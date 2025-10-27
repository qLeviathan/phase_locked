//! ZORDIC Attention Mechanism
//! Index-distance based attention without matrix multiplication

use phi_core::zordic::{IndexSet, Zordic, ZordicTensor};

/// ZORDIC attention configuration
pub struct AttentionConfig {
    pub num_heads: usize,
    pub shells_per_head: usize,
}

/// ZORDIC attention mechanism
pub struct ZordicAttention {
    zordic: Zordic,
    config: AttentionConfig,
}

impl ZordicAttention {
    pub fn new(config: AttentionConfig) -> Self {
        Self {
            zordic: Zordic::new(),
            config,
        }
    }
    
    /// Compute attention scores between query and key positions
    /// Returns negative distance (closer = higher score)
    pub fn compute_scores(&self, query: &IndexSet, keys: &[IndexSet]) -> Vec<i64> {
        keys.iter()
            .map(|key| -(self.zordic.distance(query, key) as i64))
            .collect()
    }
    
    /// Winner-take-all selection (no softmax!)
    pub fn select_winner(&self, scores: &[i64]) -> usize {
        scores.iter()
            .enumerate()
            .max_by_key(|(_, &score)| score)
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
    
    /// Single-head attention
    pub fn attention_single_head(
        &self,
        queries: &[IndexSet],
        keys: &[IndexSet],
        values: &[IndexSet],
    ) -> Vec<IndexSet> {
        queries.iter()
            .map(|query| {
                let scores = self.compute_scores(query, keys);
                let winner_idx = self.select_winner(&scores);
                values[winner_idx].clone()
            })
            .collect()
    }
    
    /// Multi-head attention via shell partitioning
    pub fn attention_multi_head(
        &self,
        tensor: &ZordicTensor,
        batch: usize,
    ) -> ZordicTensor {
        let seq_len = tensor.seq_len;
        let mut output = ZordicTensor::new(1, seq_len, 3);
        
        // For each head, process a range of shells
        for head in 0..self.config.num_heads {
            let shell_start = head * self.config.shells_per_head;
            let shell_end = (head + 1) * self.config.shells_per_head;
            
            // Extract head-specific indices for each position
            let mut head_queries = Vec::new();
            let mut head_keys = Vec::new();
            let mut head_values = Vec::new();
            
            for s in 0..seq_len {
                let indices = tensor.get(batch, s, 0); // φ-component
                let mut head_indices = IndexSet::new();
                
                // Filter indices to this head's shell range
                for &k in &indices.indices {
                    if k >= shell_start as u8 && k < shell_end as u8 {
                        head_indices.insert(k - shell_start as u8);
                    }
                }
                
                head_queries.push(head_indices.clone());
                head_keys.push(head_indices.clone());
                head_values.push(head_indices);
            }
            
            // Apply attention for this head
            let head_output = self.attention_single_head(&head_queries, &head_keys, &head_values);
            
            // Merge head outputs back (shift indices back to original range)
            for (s, indices) in head_output.into_iter().enumerate() {
                for &k in &indices.indices {
                    output.get_mut(0, s, 0).insert(k + shell_start as u8);
                }
            }
        }
        
        // Cascade final output to resolve violations
        for s in 0..seq_len {
            self.zordic.cascade(output.get_mut(0, s, 0));
        }
        
        output
    }
}

/// Causal masking for autoregressive generation
pub struct CausalMask;

impl CausalMask {
    /// Apply causal mask by restricting key positions
    pub fn apply(query_pos: usize, key_positions: &[usize]) -> Vec<usize> {
        key_positions
            .iter()
            .enumerate()
            .filter(|(_, &pos)| pos <= query_pos)
            .map(|(idx, _)| idx)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_attention_scores() {
        let config = AttentionConfig {
            num_heads: 4,
            shells_per_head: 8,
        };
        let attention = ZordicAttention::new(config);
        
        let query = IndexSet::from_indices(vec![0, 2, 5]);
        let keys = vec![
            IndexSet::from_indices(vec![0, 2, 5]), // Same as query
            IndexSet::from_indices(vec![1, 3, 6]), // Different
            IndexSet::from_indices(vec![0, 5, 6]), // Partial overlap
        ];
        
        let scores = attention.compute_scores(&query, &keys);
        
        // Score to self should be highest (0 distance)
        assert_eq!(attention.select_winner(&scores), 0);
    }
    
    #[test]
    fn test_causal_mask() {
        let positions = vec![0, 1, 2, 3, 4];
        let masked = CausalMask::apply(2, &positions);
        assert_eq!(masked, vec![0, 1, 2]); // Can only attend to positions <= 2
    }
}