// =============================================================================
// PHI_CONTEXT - Context Window Dynamics
// =============================================================================
// The context state evolves through the recurrence:
//
//   C_{n+1} = C_n ⊕ token_n
//
// Where ⊕ is superposition + cascade (the MERGE operator).
//
// The cascade_count during each fold IS the attention weight for that token.
// Tokens that cause more cascades are more "resonant" with context.
//
// Inference Path:
//   1. Reset context to zero
//   2. For each token in sequence:
//      a. Encode token → Zeckbits
//      b. Fold: C = C ⊕ token_zeck
//      c. Record cascade_count (attention weight)
//   3. Final C is the context representation
//
// The window can slide by "unfolding" old tokens (reverse operation)
// or by simply resetting and reprocessing (simpler, deterministic).
// =============================================================================

module phi_context #(
    parameter N = 32,           // Number of shells
    parameter WIDTH = 32,       // Token value width
    parameter MAX_SEQ = 64      // Maximum sequence length
)(
    input  wire              clk,
    input  wire              rst_n,

    // Control
    input  wire              clear,         // Reset context to zero
    input  wire              fold_start,    // Start folding a token

    // Token input (either pre-encoded or raw)
    input  wire [N-1:0]      token_zeck,    // Pre-encoded Zeckbits

    // Context state output
    output reg  [N-1:0]      context_state, // Current accumulated context
    output wire              fold_done,     // Fold operation complete
    output wire [7:0]        fold_norm,     // Cascade count from last fold

    // Statistics
    output reg  [7:0]        seq_length,    // Number of tokens folded
    output reg  [15:0]       total_cascades // Total cascades in context build
);

    // Internal wiring
    wire [N-1:0] merged_out;
    wire         merge_done;
    wire [7:0]   merge_norm;

    // State machine
    reg [1:0] state;
    localparam IDLE    = 2'd0;
    localparam FOLDING = 2'd1;
    localparam UPDATE  = 2'd2;

    reg fold_trigger;

    // Merge instance: context_state ⊕ token_zeck
    zeck_merge #(.N(N)) folder (
        .clk(clk),
        .rst_n(rst_n),
        .start(fold_trigger),
        .A(context_state),
        .B(token_zeck),
        .Z(merged_out),
        .done(merge_done),
        .norm_proxy(merge_norm)
    );

    assign fold_done = (state == UPDATE);
    assign fold_norm = merge_norm;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            context_state <= {N{1'b0}};
            seq_length <= 8'd0;
            total_cascades <= 16'd0;
            fold_trigger <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    fold_trigger <= 1'b0;

                    if (clear) begin
                        // Reset context
                        context_state <= {N{1'b0}};
                        seq_length <= 8'd0;
                        total_cascades <= 16'd0;
                    end else if (fold_start) begin
                        // Start folding token into context
                        fold_trigger <= 1'b1;
                        state <= FOLDING;
                    end
                end

                FOLDING: begin
                    fold_trigger <= 1'b0;

                    if (merge_done) begin
                        // Update context state
                        context_state <= merged_out;
                        seq_length <= seq_length + 8'd1;
                        total_cascades <= total_cascades + {8'd0, merge_norm};
                        state <= UPDATE;
                    end
                end

                UPDATE: begin
                    // Single cycle update pulse, return to idle
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
