// =============================================================================
// ZECK_TOP - φ-Subscript Calculus Primitives
// =============================================================================
// Complete system for Verilator testing on Raspberry Pi / Kano
//
// Primitives:
//   1. ENCODE:   token → Σφₙᵢ (Zeckendorf decomposition)
//   2. CASCADE:  rewrite physics (R1: merge, R2: split)
//   3. MERGE:    A ⊕ B = Normalize(A ⊞ B)
//   4. CONTRACT: ⟨Q, K⟩ = Σ L_{|a-b|} (Lucas attention)
//   5. CONTEXT:  Recurrence dynamics (C_{n+1} = C_n ⊕ token_n)
//   6. DECODE:   Σφₙᵢ → integer
//   7. INFER:    Complete inference FSM
//
// The cascade_count IS the Z[φ] norm. Geometry determines attention.
// =============================================================================

module zeck_top #(
    parameter N = 32,
    parameter WIDTH = 32
)(
    input  wire              clk,
    input  wire              rst_n,

    // =========================================================================
    // Encode interface: integer → Zeckbits
    // =========================================================================
    input  wire              encode_start,
    input  wire [WIDTH-1:0]  encode_value,
    output wire [N-1:0]      encode_zeck,
    output wire              encode_done,

    // =========================================================================
    // Cascade interface: rewrite to canonical form
    // =========================================================================
    input  wire              cascade_start,
    input  wire [N-1:0]      cascade_in,
    output wire [N-1:0]      cascade_out,
    output wire              cascade_done,
    output wire [7:0]        cascade_count,

    // =========================================================================
    // Merge interface: superposition + cascade
    // =========================================================================
    input  wire              merge_start,
    input  wire [N-1:0]      merge_a,
    input  wire [N-1:0]      merge_b,
    output wire [N-1:0]      merge_out,
    output wire              merge_done,
    output wire [7:0]        merge_norm,

    // =========================================================================
    // Contract interface: Lucas attention
    // =========================================================================
    input  wire              contract_start,
    input  wire [N-1:0]      contract_q,
    input  wire [N-1:0]      contract_k,
    output wire [31:0]       contract_attention,
    output wire              contract_done,

    // =========================================================================
    // Context interface: recurrence dynamics
    // =========================================================================
    input  wire              context_clear,
    input  wire              context_fold_start,
    input  wire [N-1:0]      context_token_zeck,
    output wire [N-1:0]      context_state,
    output wire              context_fold_done,
    output wire [7:0]        context_fold_norm,
    output wire [7:0]        context_seq_length,
    output wire [15:0]       context_total_cascades,

    // =========================================================================
    // Decode interface: Zeckbits → integer
    // =========================================================================
    input  wire              decode_start,
    input  wire [N-1:0]      decode_zeck_in,
    output wire [WIDTH-1:0]  decode_value,
    output wire              decode_done,

    // =========================================================================
    // Inference interface: complete FSM
    // =========================================================================
    input  wire              infer_start,
    input  wire              infer_reset_context,
    input  wire [WIDTH-1:0]  infer_token_in,
    input  wire              infer_attend_enable,
    output wire [N-1:0]      infer_context_out,
    output wire [5:0]        infer_prediction_idx,
    output wire [WIDTH-1:0]  infer_prediction_val,
    output wire [31:0]       infer_attention_sum,
    output wire              infer_done,
    output wire [7:0]        infer_step_cascades,
    output wire [7:0]        infer_seq_pos
);

    // -------------------------------------------------------------------------
    // Encoder: token → Σφₙᵢ
    // -------------------------------------------------------------------------
    zeck_encode #(.N(N), .WIDTH(WIDTH)) encoder (
        .clk(clk),
        .rst_n(rst_n),
        .start(encode_start),
        .value_in(encode_value),
        .zeck_out(encode_zeck),
        .done(encode_done)
    );

    // -------------------------------------------------------------------------
    // Cascade: rewrite engine (R1 + R2)
    // -------------------------------------------------------------------------
    zeck_cascade #(.N(N)) cascader (
        .clk(clk),
        .rst_n(rst_n),
        .start(cascade_start),
        .zeck_in(cascade_in),
        .zeck_out(cascade_out),
        .done(cascade_done),
        .cascade_count(cascade_count)
    );

    // -------------------------------------------------------------------------
    // Merge: A ⊕ B = Normalize(A ⊞ B)
    // -------------------------------------------------------------------------
    zeck_merge #(.N(N)) merger (
        .clk(clk),
        .rst_n(rst_n),
        .start(merge_start),
        .A(merge_a),
        .B(merge_b),
        .Z(merge_out),
        .done(merge_done),
        .norm_proxy(merge_norm)
    );

    // -------------------------------------------------------------------------
    // Contract: ⟨Q, K⟩ = Σ L_{|a-b|}
    // -------------------------------------------------------------------------
    phi_contract #(.N(N)) contractor (
        .clk(clk),
        .rst_n(rst_n),
        .start(contract_start),
        .Q(contract_q),
        .K(contract_k),
        .attention(contract_attention),
        .done(contract_done)
    );

    // -------------------------------------------------------------------------
    // Context: C_{n+1} = C_n ⊕ token_n
    // -------------------------------------------------------------------------
    phi_context #(.N(N), .WIDTH(WIDTH)) context_unit (
        .clk(clk),
        .rst_n(rst_n),
        .clear(context_clear),
        .fold_start(context_fold_start),
        .token_zeck(context_token_zeck),
        .context_state(context_state),
        .fold_done(context_fold_done),
        .fold_norm(context_fold_norm),
        .seq_length(context_seq_length),
        .total_cascades(context_total_cascades)
    );

    // -------------------------------------------------------------------------
    // Decode: Σφₙᵢ → integer
    // -------------------------------------------------------------------------
    phi_decode #(.N(N), .WIDTH(WIDTH)) decoder (
        .clk(clk),
        .rst_n(rst_n),
        .start(decode_start),
        .zeck_in(decode_zeck_in),
        .value_out(decode_value),
        .done(decode_done)
    );

    // -------------------------------------------------------------------------
    // Inference FSM: complete inference pipeline
    // Note: Has its own internal encoder/cascade/decoder for standalone use
    // -------------------------------------------------------------------------
    phi_infer #(.N(N), .WIDTH(WIDTH)) inferrer (
        .clk(clk),
        .rst_n(rst_n),
        .start(infer_start),
        .reset_context(infer_reset_context),
        .token_in(infer_token_in),
        .attend_enable(infer_attend_enable),
        .context_out(infer_context_out),
        .prediction_idx(infer_prediction_idx),
        .prediction_val(infer_prediction_val),
        .attention_sum(infer_attention_sum),
        .done(infer_done),
        .step_cascades(infer_step_cascades),
        .seq_pos(infer_seq_pos)
    );

endmodule
