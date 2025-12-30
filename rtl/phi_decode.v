// =============================================================================
// PHI_DECODE - Zeckendorf to Integer Decoder
// =============================================================================
// Convert shell occupancy (Zeckbits) back to integer value.
//
// value = Σᵢ zeck[i] × F_{i+2}
//
// This is the inverse of zeck_encode.
// Used to produce final token predictions from context state.
// =============================================================================

module phi_decode #(
    parameter N = 32,
    parameter WIDTH = 32
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              start,
    input  wire [N-1:0]      zeck_in,       // Shell occupancy
    output reg  [WIDTH-1:0]  value_out,     // Decoded integer
    output reg               done
);

    // Fibonacci LUT (same as encoder)
    reg [WIDTH-1:0] FIB [0:N-1];

    initial begin
        FIB[0]  = 32'd1;      // F_2
        FIB[1]  = 32'd2;      // F_3
        FIB[2]  = 32'd3;      // F_4
        FIB[3]  = 32'd5;      // F_5
        FIB[4]  = 32'd8;      // F_6
        FIB[5]  = 32'd13;     // F_7
        FIB[6]  = 32'd21;     // F_8
        FIB[7]  = 32'd34;     // F_9
        FIB[8]  = 32'd55;     // F_10
        FIB[9]  = 32'd89;     // F_11
        FIB[10] = 32'd144;    // F_12
        FIB[11] = 32'd233;    // F_13
        FIB[12] = 32'd377;    // F_14
        FIB[13] = 32'd610;    // F_15
        FIB[14] = 32'd987;    // F_16
        FIB[15] = 32'd1597;   // F_17
        FIB[16] = 32'd2584;   // F_18
        FIB[17] = 32'd4181;   // F_19
        FIB[18] = 32'd6765;   // F_20
        FIB[19] = 32'd10946;  // F_21
        FIB[20] = 32'd17711;  // F_22
        FIB[21] = 32'd28657;  // F_23
        FIB[22] = 32'd46368;  // F_24
        FIB[23] = 32'd75025;  // F_25
        FIB[24] = 32'd121393; // F_26
        FIB[25] = 32'd196418; // F_27
        FIB[26] = 32'd317811; // F_28
        FIB[27] = 32'd514229; // F_29
        FIB[28] = 32'd832040; // F_30
        FIB[29] = 32'd1346269;// F_31
        FIB[30] = 32'd2178309;// F_32
        FIB[31] = 32'd3524578;// F_33
    end

    reg [5:0] idx;
    reg [WIDTH-1:0] acc;
    reg [1:0] state;

    localparam IDLE = 2'd0;
    localparam SUM  = 2'd1;
    localparam DONE = 2'd2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            value_out <= {WIDTH{1'b0}};
            idx <= 6'd0;
            acc <= {WIDTH{1'b0}};
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        idx <= 6'd0;
                        acc <= {WIDTH{1'b0}};
                        state <= SUM;
                    end
                end

                SUM: begin
                    if (idx < N) begin
                        if (zeck_in[idx]) begin
                            acc <= acc + FIB[idx];
                        end
                        idx <= idx + 6'd1;
                    end else begin
                        state <= DONE;
                    end
                end

                DONE: begin
                    value_out <= acc;
                    done <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
