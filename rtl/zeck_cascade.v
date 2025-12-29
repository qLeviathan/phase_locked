// =============================================================================
// ZECKBIT CASCADE CORE - Pure Rewrite Physics
// =============================================================================
// Target: Verilator on Raspberry Pi 3B
//
// This is NOT binary arithmetic. This is a constraint-driven rewrite automaton.
//
// The "bit" is Fibonacci-shell occupancy, not base-2 weight.
// The cascade implements F_k + F_{k+1} = F_{k+2} (merge)
// And 2*F_k = F_{k+1} + F_{k-2} (split)
//
// NO BOOLEAN XOR. The operator is: superposition → cascade → canonical form
// =============================================================================

module zeck_cascade #(
    parameter N = 32  // Number of shells (Fibonacci indices)
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [N-1:0] zeck_in,    // Input zeckbits (shell occupancy)
    output reg  [N-1:0] zeck_out,   // Normalized output (canonical form)
    output reg         done,
    output reg  [7:0]  cascade_count // How many rewrites occurred (= "work" = norm proxy)
);

    // Internal state: ternary digits 0,1,2 (2 bits each)
    // D[i] = 0: shell empty
    // D[i] = 1: shell occupied
    // D[i] = 2: shell double-occupied (needs R2 split)
    reg [1:0] D [0:N-1];

    reg [5:0] i;           // Scan pointer
    reg       changed;     // Did this pass change anything?
    reg [1:0] state;

    localparam IDLE   = 2'b00;
    localparam LOAD   = 2'b01;
    localparam RUN    = 2'b10;
    localparam FINISH = 2'b11;

    integer k;

    // Saturating increment (clamp at 2)
    function [1:0] inc_sat;
        input [1:0] x;
        begin
            inc_sat = (x == 2'd2) ? 2'd2 : x + 2'd1;
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            cascade_count <= 8'd0;
            changed <= 1'b0;
            i <= 6'd0;
            zeck_out <= {N{1'b0}};
            for (k = 0; k < N; k = k + 1) begin
                D[k] <= 2'd0;
            end
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        state <= LOAD;
                    end
                end

                LOAD: begin
                    // Load input zeckbits into ternary digits
                    for (k = 0; k < N; k = k + 1) begin
                        D[k] <= {1'b0, zeck_in[k]};
                    end
                    i <= 6'd0;
                    changed <= 1'b0;
                    cascade_count <= 8'd0;
                    state <= RUN;
                end

                RUN: begin
                    if (i < N) begin
                        // =============================================
                        // RULE R2 (split): 2@i → 1@(i+1) + 1@(i-2)
                        // Priority: handle double-occupancy first
                        // =============================================
                        if ((D[i] == 2'd2) && (i >= 2) && (i + 1 < N)) begin
                            D[i]   <= 2'd0;
                            D[i+1] <= inc_sat(D[i+1]);
                            D[i-2] <= inc_sat(D[i-2]);
                            changed <= 1'b1;
                            cascade_count <= cascade_count + 8'd1;
                            // Backtrack to check for new violations
                            i <= (i >= 2) ? i - 2 : 6'd0;
                        end
                        // =============================================
                        // RULE R1 (merge): 11@(i,i+1) → 001@(i,i+1,i+2)
                        // Adjacent shells collapse upward
                        // =============================================
                        else if ((i + 2 < N) && (D[i] != 2'd0) && (D[i+1] != 2'd0)) begin
                            // Consume one from each adjacent shell
                            D[i]   <= D[i] - 2'd1;
                            D[i+1] <= D[i+1] - 2'd1;
                            // Produce one at i+2
                            D[i+2] <= inc_sat(D[i+2]);
                            changed <= 1'b1;
                            cascade_count <= cascade_count + 8'd1;
                            // Backtrack
                            i <= (i > 0) ? i - 1 : 6'd0;
                        end
                        else begin
                            i <= i + 6'd1;
                        end
                    end else begin
                        // End of pass
                        if (changed) begin
                            // More work needed, start new pass
                            i <= 6'd0;
                            changed <= 1'b0;
                        end else begin
                            // Stable: canonical form reached
                            state <= FINISH;
                        end
                    end
                end

                FINISH: begin
                    // Output canonical zeckbits (D[i] should be 0 or 1 now)
                    for (k = 0; k < N; k = k + 1) begin
                        zeck_out[k] <= D[k][0];
                    end
                    done <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
