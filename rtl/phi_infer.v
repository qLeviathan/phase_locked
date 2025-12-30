// =============================================================================
// PHI_INFER - Complete Inference FSM
// =============================================================================
// The φ-mechanics inference state machine.
//
// States:
//   IDLE   - Waiting for input
//   ENCODE - Convert token to Zeckbits (via zeck_encode)
//   FOLD   - Merge token into context (C = C ⊕ token)
//   SNAP   - Find highest occupied shell (nearest cell)
//   DECODE - Convert prediction back to token integer
//   OUTPUT - Present result
//
// Inference Semantics:
//   - The context state C accumulates tokens via folding
//   - The highest occupied shell = dominant frequency = prediction
//   - The step_cascades count IS the attention weight for this token
//
// Key Insight:
//   The cascade_count during fold IS computing distance on the hyperbola.
//   More cascades = more work = shells farther apart = less similar.
//   Fewer cascades = shells overlapping well = high similarity.
// =============================================================================

module phi_infer #(
    parameter N = 32,
    parameter WIDTH = 32
)(
    input  wire              clk,
    input  wire              rst_n,

    // Control
    input  wire              start,          // Start inference step
    input  wire              reset_context,  // Clear context to zero

    // Input token
    input  wire [WIDTH-1:0]  token_in,       // Raw integer token

    // Attention control (simplified)
    input  wire              attend_enable,  // Reserved for future use

    // Outputs
    output reg  [N-1:0]      context_out,    // Current context state
    output reg  [5:0]        prediction_idx, // Highest occupied shell index
    output reg  [WIDTH-1:0]  prediction_val, // Decoded prediction value
    output reg  [31:0]       attention_sum,  // Cascade count = attention
    output reg               done,

    // Statistics
    output reg  [7:0]        step_cascades,  // Cascades in this step
    output reg  [7:0]        seq_pos         // Current sequence position
);

    // =========================================================================
    // State Machine
    // =========================================================================
    reg [2:0] state;
    localparam IDLE   = 3'd0;
    localparam ENCODE = 3'd1;
    localparam FOLD   = 3'd2;
    localparam SNAP   = 3'd3;
    localparam DECODE = 3'd4;
    localparam OUTPUT = 3'd5;

    // =========================================================================
    // Submodule Wiring
    // =========================================================================

    // Encoder
    reg encode_start;
    wire [N-1:0] encoded_token;
    wire encode_done;

    zeck_encode #(.N(N), .WIDTH(WIDTH)) encoder (
        .clk(clk),
        .rst_n(rst_n),
        .start(encode_start),
        .value_in(token_in),
        .zeck_out(encoded_token),
        .done(encode_done)
    );

    // Cascade (for merge)
    reg cascade_start;
    wire [N-1:0] merged_out;
    wire cascade_done;
    wire [7:0] cascade_count;

    // Superposition: context ⊞ token
    reg [N-1:0] context_state;
    wire [N-1:0] superposed = context_state | encoded_token;

    zeck_cascade #(.N(N)) merger (
        .clk(clk),
        .rst_n(rst_n),
        .start(cascade_start),
        .zeck_in(superposed),
        .zeck_out(merged_out),
        .done(cascade_done),
        .cascade_count(cascade_count)
    );

    // Decoder
    reg decode_start;
    wire [WIDTH-1:0] decoded_val;
    wire decode_done;

    phi_decode #(.N(N), .WIDTH(WIDTH)) decoder (
        .clk(clk),
        .rst_n(rst_n),
        .start(decode_start),
        .zeck_in(context_state),
        .value_out(decoded_val),
        .done(decode_done)
    );

    // =========================================================================
    // Find Highest Shell (combinational)
    // =========================================================================
    reg [5:0] highest_shell;
    integer i;

    always @(*) begin
        highest_shell = 6'd0;
        for (i = 0; i < N; i = i + 1) begin
            if (context_state[i]) begin
                highest_shell = i[5:0];
            end
        end
    end

    // =========================================================================
    // Main FSM
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            context_state <= {N{1'b0}};
            context_out <= {N{1'b0}};
            prediction_idx <= 6'd0;
            prediction_val <= {WIDTH{1'b0}};
            attention_sum <= 32'd0;
            step_cascades <= 8'd0;
            seq_pos <= 8'd0;
            encode_start <= 1'b0;
            cascade_start <= 1'b0;
            decode_start <= 1'b0;
        end else begin
            case (state)
                // ---------------------------------------------------------
                IDLE: begin
                    done <= 1'b0;
                    encode_start <= 1'b0;
                    cascade_start <= 1'b0;
                    decode_start <= 1'b0;

                    if (reset_context) begin
                        context_state <= {N{1'b0}};
                        seq_pos <= 8'd0;
                        attention_sum <= 32'd0;
                    end else if (start) begin
                        // Begin inference step
                        encode_start <= 1'b1;
                        state <= ENCODE;
                    end
                end

                // ---------------------------------------------------------
                ENCODE: begin
                    encode_start <= 1'b0;

                    if (encode_done) begin
                        // Token encoded, now fold into context
                        cascade_start <= 1'b1;
                        state <= FOLD;
                    end
                end

                // ---------------------------------------------------------
                FOLD: begin
                    cascade_start <= 1'b0;

                    if (cascade_done) begin
                        // Update context with merged result
                        context_state <= merged_out;
                        step_cascades <= cascade_count;
                        // The cascade_count IS the attention weight!
                        // Accumulate it as the total attention metric
                        attention_sum <= attention_sum + {24'd0, cascade_count};
                        seq_pos <= seq_pos + 8'd1;
                        state <= SNAP;
                    end
                end

                // ---------------------------------------------------------
                SNAP: begin
                    // Find highest occupied shell
                    prediction_idx <= highest_shell;
                    decode_start <= 1'b1;
                    state <= DECODE;
                end

                // ---------------------------------------------------------
                DECODE: begin
                    decode_start <= 1'b0;

                    if (decode_done) begin
                        prediction_val <= decoded_val;
                        state <= OUTPUT;
                    end
                end

                // ---------------------------------------------------------
                OUTPUT: begin
                    context_out <= context_state;
                    done <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
