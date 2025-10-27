//! Holographic Memory System
//! Content-addressable memory indexed by Ω values

use std::collections::{HashMap, VecDeque};
use phi_core::zordic::{IndexSet, Zordic};

/// Memory entry with pattern and metadata
#[derive(Clone, Debug)]
pub struct MemoryEntry {
    pub pattern: IndexSet,
    pub omega: u64,
    pub frequency: u32,
    pub timestamp: u64,
    pub context: Option<String>,
}

/// Holographic memory with Ω-based indexing
pub struct HolographicMemory {
    /// Main index: Ω value → list of patterns
    omega_index: HashMap<u64, Vec<MemoryEntry>>,
    
    /// Reverse index for fast pattern lookup
    pattern_index: HashMap<Vec<u8>, u64>,
    
    /// Configuration
    max_entries_per_omega: usize,
    capacity: usize,
    current_size: usize,
    timestamp: u64,
    
    /// For similarity search
    tolerance: u32,
    
    /// Zordic core for operations
    zordic: Zordic,
}

impl HolographicMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            omega_index: HashMap::new(),
            pattern_index: HashMap::new(),
            max_entries_per_omega: 10,
            capacity,
            current_size: 0,
            timestamp: 0,
            tolerance: 3,
            zordic: Zordic::new(),
        }
    }
    
    /// Store a pattern in memory
    pub fn store(&mut self, pattern: IndexSet, context: Option<String>) -> u64 {
        let omega = pattern.omega(&self.zordic.fib);
        let pattern_vec = pattern.to_vec();
        
        // Check if pattern already exists
        if let Some(&existing_omega) = self.pattern_index.get(&pattern_vec) {
            // Update frequency for existing pattern
            if let Some(entries) = self.omega_index.get_mut(&existing_omega) {
                for entry in entries.iter_mut() {
                    if entry.pattern.to_vec() == pattern_vec {
                        entry.frequency += 1;
                        entry.timestamp = self.timestamp;
                        self.timestamp += 1;
                        return omega;
                    }
                }
            }
        }
        
        // Create new entry
        let entry = MemoryEntry {
            pattern: pattern.clone(),
            omega,
            frequency: 1,
            timestamp: self.timestamp,
            context,
        };
        
        self.timestamp += 1;
        
        // Store in omega index
        self.omega_index
            .entry(omega)
            .or_insert_with(Vec::new)
            .push(entry);
        
        // Store in pattern index
        self.pattern_index.insert(pattern_vec, omega);
        
        self.current_size += 1;
        
        // Evict if over capacity
        if self.current_size > self.capacity {
            self.evict_lru();
        }
        
        omega
    }
    
    /// Retrieve exact pattern by Ω value
    pub fn retrieve(&self, omega: u64) -> Option<&Vec<MemoryEntry>> {
        self.omega_index.get(&omega)
    }
    
    /// Retrieve patterns within tolerance (Hamming distance)
    pub fn retrieve_similar(&self, pattern: &IndexSet, tolerance: u32) -> Vec<&MemoryEntry> {
        let target_omega = pattern.omega(&self.zordic.fib);
        let mut results = Vec::new();
        
        // Search within Ω range
        let omega_min = target_omega.saturating_sub(tolerance as u64 * 100);
        let omega_max = target_omega.saturating_add(tolerance as u64 * 100);
        
        for (&omega, entries) in &self.omega_index {
            if omega >= omega_min && omega <= omega_max {
                for entry in entries {
                    let distance = self.zordic.distance(pattern, &entry.pattern);
                    if distance <= tolerance as u64 {
                        results.push(entry);
                    }
                }
            }
        }
        
        // Sort by distance (closest first)
        results.sort_by_key(|entry| self.zordic.distance(pattern, &entry.pattern));
        
        results
    }
    
    /// Semantic search by context
    pub fn search_by_context(&self, query: &str) -> Vec<&MemoryEntry> {
        let mut results = Vec::new();
        
        for entries in self.omega_index.values() {
            for entry in entries {
                if let Some(ref context) = entry.context {
                    if context.contains(query) {
                        results.push(entry);
                    }
                }
            }
        }
        
        // Sort by frequency (most frequent first)
        results.sort_by_key(|entry| std::cmp::Reverse(entry.frequency));
        
        results
    }
    
    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            total_entries: self.current_size,
            unique_omegas: self.omega_index.len(),
            total_patterns: self.pattern_index.len(),
            avg_entries_per_omega: if self.omega_index.is_empty() {
                0.0
            } else {
                self.current_size as f64 / self.omega_index.len() as f64
            },
        }
    }
    
    /// Evict least recently used entries
    fn evict_lru(&mut self) {
        // Find oldest entry
        let mut oldest_omega = 0u64;
        let mut oldest_timestamp = u64::MAX;
        let mut oldest_idx = 0;
        
        for (&omega, entries) in &self.omega_index {
            for (idx, entry) in entries.iter().enumerate() {
                if entry.timestamp < oldest_timestamp {
                    oldest_timestamp = entry.timestamp;
                    oldest_omega = omega;
                    oldest_idx = idx;
                }
            }
        }
        
        // Remove oldest entry
        if let Some(entries) = self.omega_index.get_mut(&oldest_omega) {
            if oldest_idx < entries.len() {
                let removed = entries.remove(oldest_idx);
                self.pattern_index.remove(&removed.pattern.to_vec());
                self.current_size -= 1;
                
                // Remove omega entry if empty
                if entries.is_empty() {
                    self.omega_index.remove(&oldest_omega);
                }
            }
        }
    }
}

/// Memory statistics
#[derive(Debug)]
pub struct MemoryStats {
    pub total_entries: usize,
    pub unique_omegas: usize,
    pub total_patterns: usize,
    pub avg_entries_per_omega: f64,
}

/// Batch operations for efficient memory updates
impl HolographicMemory {
    /// Store multiple patterns at once
    pub fn store_batch(&mut self, patterns: Vec<(IndexSet, Option<String>)>) -> Vec<u64> {
        patterns.into_iter()
            .map(|(pattern, context)| self.store(pattern, context))
            .collect()
    }
    
    /// Clear all memory
    pub fn clear(&mut self) {
        self.omega_index.clear();
        self.pattern_index.clear();
        self.current_size = 0;
        self.timestamp = 0;
    }
    
    /// Save memory to disk (serialization)
    pub fn save(&self, path: &str) -> Result<(), std::io::Error> {
        use std::fs::File;
        use std::io::Write;
        
        let mut file = File::create(path)?;
        
        // Simple format: omega,pattern_indices,frequency,context
        for (omega, entries) in &self.omega_index {
            for entry in entries {
                let indices = entry.pattern.to_vec();
                let indices_str = indices.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                
                let context = entry.context.as_deref().unwrap_or("");
                writeln!(file, "{}|{}|{}|{}", omega, indices_str, entry.frequency, context)?;
            }
        }
        
        Ok(())
    }
    
    /// Load memory from disk
    pub fn load(&mut self, path: &str) -> Result<(), std::io::Error> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        
        self.clear();
        
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split('|').collect();
            
            if parts.len() >= 3 {
                let omega: u64 = parts[0].parse().unwrap_or(0);
                let indices: Vec<u8> = parts[1].split(',')
                    .filter_map(|s| s.parse().ok())
                    .collect();
                let frequency: u32 = parts[2].parse().unwrap_or(1);
                let context = if parts.len() > 3 && !parts[3].is_empty() {
                    Some(parts[3].to_string())
                } else {
                    None
                };
                
                let pattern = IndexSet::from_indices(indices);
                
                let entry = MemoryEntry {
                    pattern,
                    omega,
                    frequency,
                    timestamp: self.timestamp,
                    context,
                };
                
                self.timestamp += 1;
                
                self.omega_index
                    .entry(omega)
                    .or_insert_with(Vec::new)
                    .push(entry);
                
                self.current_size += 1;
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_holographic_memory() {
        let mut memory = HolographicMemory::new(100);
        
        // Store some patterns
        let pattern1 = IndexSet::from_indices(vec![0, 2, 5]);
        let pattern2 = IndexSet::from_indices(vec![1, 3, 6]);
        let pattern3 = IndexSet::from_indices(vec![0, 2, 6]); // Similar to pattern1
        
        let omega1 = memory.store(pattern1.clone(), Some("first pattern".to_string()));
        let omega2 = memory.store(pattern2.clone(), Some("second pattern".to_string()));
        let omega3 = memory.store(pattern3.clone(), Some("similar to first".to_string()));
        
        // Test exact retrieval
        assert!(memory.retrieve(omega1).is_some());
        assert_eq!(memory.retrieve(omega1).unwrap().len(), 1);
        
        // Test similarity search
        let similar = memory.retrieve_similar(&pattern1, 3);
        assert!(similar.len() >= 2); // Should find pattern1 and pattern3
        
        // Test context search
        let context_results = memory.search_by_context("first");
        assert_eq!(context_results.len(), 2);
        
        // Test stats
        let stats = memory.stats();
        assert_eq!(stats.total_entries, 3);
        println!("Memory stats: {:?}", stats);
    }
}