// =============================================================================
// ZECK_TOP - Top Level for Verilator Testing
// =============================================================================
// Demonstrates the core cascade primitive on Raspberry Pi
//
// Operations:
//   1. Encode integer → zeckbits
//   2. Cascade normalize (the physics)
//   3. Merge two states (superposition → cascade)
//
// The cascade_count is the key output: it measures "work" to reach
// canonical form, which IS the Z[φ] norm (shell metric).
// =============================================================================

module zeck_top #(
    parameter N = 32,
    parameter WIDTH = 32
)(
    input  wire              clk,
    input  wire              rst_n,

    // Encode interface
    input  wire              encode_start,
    input  wire [WIDTH-1:0]  encode_value,
    output wire [N-1:0]      encode_zeck,
    output wire              encode_done,

    // Raw cascade interface (for testing rewrite physics)
    input  wire              cascade_start,
    input  wire [N-1:0]      cascade_in,
    output wire [N-1:0]      cascade_out,
    output wire              cascade_done,
    output wire [7:0]        cascade_count,

    // Merge interface (the true "XOR" = superposition + cascade)
    input  wire              merge_start,
    input  wire [N-1:0]      merge_a,
    input  wire [N-1:0]      merge_b,
    output wire [N-1:0]      merge_out,
    output wire              merge_done,
    output wire [7:0]        merge_norm   // Cascade count = norm proxy
);

    // Encoder instance
    zeck_encode #(.N(N), .WIDTH(WIDTH)) encoder (
        .clk(clk),
        .rst_n(rst_n),
        .start(encode_start),
        .value_in(encode_value),
        .zeck_out(encode_zeck),
        .done(encode_done)
    );

    // Cascade instance (raw rewrite engine)
    zeck_cascade #(.N(N)) cascader (
        .clk(clk),
        .rst_n(rst_n),
        .start(cascade_start),
        .zeck_in(cascade_in),
        .zeck_out(cascade_out),
        .done(cascade_done),
        .cascade_count(cascade_count)
    );

    // Merge instance (superposition + cascade)
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

endmodule
