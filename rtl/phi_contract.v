// =============================================================================
// PHI_CONTRACT - Lucas Contraction for Attention
// =============================================================================
// ⟨Q, K⟩ = Σᵢ Σⱼ L_{aᵢ - bⱼ}
//
// Where Q = Σφ_{aᵢ} and K = Σφ_{bⱼ} are Zeckendorf-encoded states.
//
// This computes attention weight between two shell states.
// The result is purely integer - geometry determines the weight.
// =============================================================================

module phi_contract #(
    parameter N = 32  // Number of shells
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [N-1:0] Q,           // Query shells (Zeckbits)
    input  wire [N-1:0] K,           // Key shells (Zeckbits)
    output reg  [31:0] attention,    // ⟨Q, K⟩ = Σ L_{|a-b|}
    output reg         done
);

    // Lucas LUT
    wire [31:0] lucas_val;
    reg [5:0] lucas_idx;

    lucas_lut #(.DEPTH(N), .WIDTH(32)) lut (
        .index(lucas_idx),
        .lucas_val(lucas_val)
    );

    // State
    reg [5:0] i, j;
    reg [31:0] acc;
    reg [1:0] state;

    localparam IDLE = 2'd0;
    localparam SCAN = 2'd1;
    localparam DONE = 2'd2;

    // Absolute difference
    function [5:0] abs_diff;
        input [5:0] a, b;
        begin
            abs_diff = (a >= b) ? (a - b) : (b - a);
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            attention <= 32'd0;
            acc <= 32'd0;
            i <= 6'd0;
            j <= 6'd0;
            lucas_idx <= 6'd0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        acc <= 32'd0;
                        i <= 6'd0;
                        j <= 6'd0;
                        state <= SCAN;
                    end
                end

                SCAN: begin
                    if (i < N) begin
                        if (j < N) begin
                            // If both shells occupied, add L_{|i-j|}
                            if (Q[i] && K[j]) begin
                                lucas_idx <= abs_diff(i, j);
                                // Accumulate (1-cycle delay for LUT)
                                acc <= acc + lucas_val;
                            end
                            j <= j + 6'd1;
                        end else begin
                            j <= 6'd0;
                            i <= i + 6'd1;
                        end
                    end else begin
                        state <= DONE;
                    end
                end

                DONE: begin
                    attention <= acc;
                    done <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
