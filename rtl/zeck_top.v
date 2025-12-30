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
//
// The cascade_count IS the Z[φ] norm. Geometry determines attention.
// =============================================================================

module zeck_top #(
    parameter N = 32,
    parameter WIDTH = 32
)(
    input  wire              clk,
    input  wire              rst_n,

    // Encode interface: integer → Zeckbits
    input  wire              encode_start,
    input  wire [WIDTH-1:0]  encode_value,
    output wire [N-1:0]      encode_zeck,
    output wire              encode_done,

    // Cascade interface: rewrite to canonical form
    input  wire              cascade_start,
    input  wire [N-1:0]      cascade_in,
    output wire [N-1:0]      cascade_out,
    output wire              cascade_done,
    output wire [7:0]        cascade_count,

    // Merge interface: superposition + cascade
    input  wire              merge_start,
    input  wire [N-1:0]      merge_a,
    input  wire [N-1:0]      merge_b,
    output wire [N-1:0]      merge_out,
    output wire              merge_done,
    output wire [7:0]        merge_norm,

    // Contract interface: Lucas attention
    input  wire              contract_start,
    input  wire [N-1:0]      contract_q,
    input  wire [N-1:0]      contract_k,
    output wire [31:0]       contract_attention,
    output wire              contract_done
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

endmodule
