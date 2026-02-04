// ============================================================================
// UPDUINO 3.1 TOP-LEVEL MODULE
// ============================================================================
// Deterministic AI Apparatus - Patent Pending
//
// Top-level integration for Lattice iCE40 UP5K FPGA on UPduino 3.1 board.
//
// Resources Available:
// - 5280 LUTs
// - 1Mb SPRAM (4x 256Kbit blocks)
// - 120Kb DPRAM
// - 8 DSP blocks (not used - integer-only design)
// - 48MHz internal oscillator
//
// Features:
// - Deterministic AI inference with zero floating point
// - Zeckendorf (base-phi) encoding
// - Phase-locked state evolution
// - Clock-defined immutable register memory
// - UART interface for host communication
// - RGB LED status indication
//
// Target: Lattice iCE40 UP5K (UPduino 3.1)
// ============================================================================

`default_nettype none
`timescale 1ns / 1ps

module upduino_top (
    // System
    input  wire         clk_48mhz,      // 48MHz oscillator (directly on die)

    // UART
    input  wire         uart_rx,        // UART receive
    output wire         uart_tx,        // UART transmit

    // RGB LED (active low on UPduino)
    output wire         led_red_n,
    output wire         led_green_n,
    output wire         led_blue_n,

    // SPI Flash (directly accessible)
    output wire         spi_cs_n,
    output wire         spi_sck,
    output wire         spi_mosi,
    input  wire         spi_miso,

    // GPIO (directly accessible)
    inout  wire [7:0]   gpio
);

// ============================================================================
// CLOCK AND RESET
// ============================================================================

// Use internal 48MHz oscillator
wire clk;
wire rst_n;

`ifdef SYNTHESIS
    // Internal oscillator
    SB_HFOSC #(
        .CLKHF_DIV("0b00")  // 48MHz
    ) osc_inst (
        .CLKHFPU(1'b1),
        .CLKHFEN(1'b1),
        .CLKHF(clk)
    );
`else
    assign clk = clk_48mhz;
`endif

// Reset generator (simple power-on reset)
reg [7:0] reset_counter = 8'd0;
reg       reset_done = 1'b0;

always @(posedge clk) begin
    if (!reset_done) begin
        reset_counter <= reset_counter + 8'd1;
        if (reset_counter == 8'hFF) begin
            reset_done <= 1'b1;
        end
    end
end

assign rst_n = reset_done;

// ============================================================================
// UART (115200 baud @ 48MHz)
// ============================================================================
localparam BAUD_DIV = 48000000 / 115200;  // = 416

// UART TX
reg  [7:0]  uart_tx_data;
reg         uart_tx_valid;
wire        uart_tx_ready;
wire        uart_tx_out;

// UART RX
wire [7:0]  uart_rx_data;
wire        uart_rx_valid;
reg         uart_rx_ready;

uart_tx #(.CLK_DIV(BAUD_DIV)) uart_tx_inst (
    .clk(clk),
    .rst_n(rst_n),
    .data(uart_tx_data),
    .valid(uart_tx_valid),
    .ready(uart_tx_ready),
    .tx(uart_tx_out)
);

uart_rx #(.CLK_DIV(BAUD_DIV)) uart_rx_inst (
    .clk(clk),
    .rst_n(rst_n),
    .rx(uart_rx),
    .data(uart_rx_data),
    .valid(uart_rx_valid),
    .ready(uart_rx_ready)
);

assign uart_tx = uart_tx_out;

// ============================================================================
// ZECKENDORF ARITHMETIC UNIT
// ============================================================================
wire        zeck_start;
wire [2:0]  zeck_opcode;
wire [31:0] zeck_operand_a;
wire [31:0] zeck_operand_b;
wire        zeck_done;
wire        zeck_valid;
wire [31:0] zeck_result;
wire [5:0]  zeck_omega;
wire [4:0]  zeck_complexity;

zeckendorf_arith #(
    .WIDTH(32),
    .FIB_DEPTH(47)
) zeck_arith_inst (
    .clk(clk),
    .rst_n(rst_n),
    .start(zeck_start),
    .opcode(zeck_opcode),
    .operand_a(zeck_operand_a),
    .operand_b(zeck_operand_b),
    .done(zeck_done),
    .valid(zeck_valid),
    .result(zeck_result),
    .omega(zeck_omega),
    .complexity(zeck_complexity)
);

// ============================================================================
// CASCADE OPERATOR
// ============================================================================
wire        cascade_start;
wire [31:0] cascade_bits_in;
wire        cascade_done;
wire        cascade_valid;
wire [31:0] cascade_bits_out;

cascade_iterative #(
    .WIDTH(32),
    .MAX_ITERATIONS(15)
) cascade_inst (
    .clk(clk),
    .rst_n(rst_n),
    .start(cascade_start),
    .bits_in(cascade_bits_in),
    .done(cascade_done),
    .valid(cascade_valid),
    .bits_out(cascade_bits_out),
    .iterations()
);

// ============================================================================
// CORDIC UNIT
// ============================================================================
wire                cordic_start;
wire signed [31:0]  cordic_angle;
wire                cordic_done;
wire signed [31:0]  cordic_sin;
wire signed [31:0]  cordic_cos;

cordic_sincos #(.WIDTH(32)) cordic_inst (
    .clk(clk),
    .rst_n(rst_n),
    .start(cordic_start),
    .angle(cordic_angle),
    .done(cordic_done),
    .sin_out(cordic_sin),
    .cos_out(cordic_cos)
);

// ============================================================================
// OMEGA-INDEXED MEMORY
// ============================================================================
wire        mem_store_valid;
wire [31:0] mem_store_zeck;
wire [31:0] mem_store_energy;
wire [31:0] mem_store_theta;
wire [5:0]  mem_store_omega;
wire        mem_store_done;

wire        mem_search_valid;
wire [5:0]  mem_search_omega;
wire        mem_search_done;
wire [3:0]  mem_search_count;
wire [31:0] mem_search_best_zeck;
wire [31:0] mem_search_best_energy;

omega_memory #(
    .ZECK_WIDTH(32),
    .ENERGY_WIDTH(32),
    .THETA_WIDTH(32),
    .OMEGA_WIDTH(6),
    .ENTRIES_PER_OMEGA(16)
) omega_mem_inst (
    .clk(clk),
    .rst_n(rst_n),

    .store_valid(mem_store_valid),
    .store_zeck(mem_store_zeck),
    .store_energy(mem_store_energy),
    .store_theta(mem_store_theta),
    .store_omega(mem_store_omega),
    .store_done(mem_store_done),

    .recall_valid(1'b0),
    .recall_omega(6'd0),
    .recall_done(),
    .recall_found(),
    .recall_zeck(),
    .recall_energy(),
    .recall_theta(),

    .search_valid(mem_search_valid),
    .search_omega(mem_search_omega),
    .search_tolerance(3'd2),
    .search_done(mem_search_done),
    .search_count(mem_search_count),
    .search_best_zeck(mem_search_best_zeck),
    .search_best_energy(mem_search_best_energy)
);

// ============================================================================
// FSM CONTROLLER
// ============================================================================
wire        fsm_start;
wire        fsm_seed_valid;
wire [31:0] fsm_seed_token;
wire        fsm_ready;
wire        fsm_halted;
wire        fsm_emit_valid;
wire [31:0] fsm_emit_token;
wire [5:0]  fsm_emit_omega;
wire [3:0]  fsm_current_state;
wire [31:0] fsm_step_count;

zeck_fsm_controller #(
    .WIDTH(32),
    .CONTEXT_DEPTH(8),
    .CANDIDATE_COUNT(8)
) fsm_ctrl_inst (
    .clk(clk),
    .rst_n(rst_n),

    .start(fsm_start),
    .seed_valid(fsm_seed_valid),
    .seed_token(fsm_seed_token),
    .ready(fsm_ready),
    .halted(fsm_halted),

    .emit_valid(fsm_emit_valid),
    .emit_token(fsm_emit_token),
    .emit_omega(fsm_emit_omega),

    .zeck_start(zeck_start),
    .zeck_opcode(zeck_opcode),
    .zeck_operand_a(zeck_operand_a),
    .zeck_operand_b(zeck_operand_b),
    .zeck_done(zeck_done),
    .zeck_valid(zeck_valid),
    .zeck_result(zeck_result),
    .zeck_omega(zeck_omega),
    .zeck_complexity(zeck_complexity),

    .cascade_start(cascade_start),
    .cascade_bits_in(cascade_bits_in),
    .cascade_done(cascade_done),
    .cascade_valid(cascade_valid),
    .cascade_bits_out(cascade_bits_out),

    .cordic_start(cordic_start),
    .cordic_angle(cordic_angle),
    .cordic_done(cordic_done),
    .cordic_sin(cordic_sin),
    .cordic_cos(cordic_cos),

    .mem_store_valid(mem_store_valid),
    .mem_store_zeck(mem_store_zeck),
    .mem_store_energy(mem_store_energy),
    .mem_store_theta(mem_store_theta),
    .mem_store_omega(mem_store_omega),
    .mem_store_done(mem_store_done),

    .mem_search_valid(mem_search_valid),
    .mem_search_omega(mem_search_omega),
    .mem_search_done(mem_search_done),
    .mem_search_count(mem_search_count),
    .mem_search_best_zeck(mem_search_best_zeck),
    .mem_search_best_energy(mem_search_best_energy),

    .current_state(fsm_current_state),
    .step_count(fsm_step_count)
);

// ============================================================================
// HOST INTERFACE (UART Command Parser)
// ============================================================================
localparam CMD_NOP    = 8'h00;
localparam CMD_RESET  = 8'h01;
localparam CMD_START  = 8'h02;
localparam CMD_SEED   = 8'h03;
localparam CMD_STATUS = 8'h04;
localparam CMD_READ   = 8'h05;

reg [2:0]  cmd_state;
reg [7:0]  cmd_byte;
reg [31:0] cmd_data;
reg [1:0]  cmd_byte_count;

localparam CS_IDLE     = 3'd0;
localparam CS_CMD      = 3'd1;
localparam CS_DATA     = 3'd2;
localparam CS_EXECUTE  = 3'd3;
localparam CS_RESPOND  = 3'd4;

reg fsm_start_reg;
reg fsm_seed_valid_reg;
reg [31:0] fsm_seed_token_reg;

assign fsm_start = fsm_start_reg;
assign fsm_seed_valid = fsm_seed_valid_reg;
assign fsm_seed_token = fsm_seed_token_reg;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cmd_state <= CS_IDLE;
        cmd_byte <= 8'd0;
        cmd_data <= 32'd0;
        cmd_byte_count <= 2'd0;
        uart_rx_ready <= 1'b0;
        uart_tx_valid <= 1'b0;
        uart_tx_data <= 8'd0;
        fsm_start_reg <= 1'b0;
        fsm_seed_valid_reg <= 1'b0;
        fsm_seed_token_reg <= 32'd0;
    end else begin
        // Clear pulses
        fsm_start_reg <= 1'b0;
        fsm_seed_valid_reg <= 1'b0;
        uart_tx_valid <= 1'b0;

        case (cmd_state)
            CS_IDLE: begin
                uart_rx_ready <= 1'b1;
                if (uart_rx_valid) begin
                    cmd_byte <= uart_rx_data;
                    cmd_state <= CS_CMD;
                end
            end

            CS_CMD: begin
                case (cmd_byte)
                    CMD_SEED: begin
                        // Expect 4 bytes of data
                        cmd_byte_count <= 2'd0;
                        cmd_state <= CS_DATA;
                    end
                    CMD_START: begin
                        cmd_state <= CS_EXECUTE;
                    end
                    CMD_STATUS: begin
                        cmd_state <= CS_RESPOND;
                    end
                    default: begin
                        cmd_state <= CS_IDLE;
                    end
                endcase
            end

            CS_DATA: begin
                uart_rx_ready <= 1'b1;
                if (uart_rx_valid) begin
                    cmd_data <= {cmd_data[23:0], uart_rx_data};
                    cmd_byte_count <= cmd_byte_count + 2'd1;
                    if (cmd_byte_count == 2'd3) begin
                        cmd_state <= CS_EXECUTE;
                    end
                end
            end

            CS_EXECUTE: begin
                case (cmd_byte)
                    CMD_START: begin
                        fsm_start_reg <= 1'b1;
                    end
                    CMD_SEED: begin
                        fsm_seed_valid_reg <= 1'b1;
                        fsm_seed_token_reg <= cmd_data;
                    end
                endcase
                cmd_state <= CS_RESPOND;
            end

            CS_RESPOND: begin
                if (uart_tx_ready) begin
                    // Send status byte
                    uart_tx_valid <= 1'b1;
                    uart_tx_data <= {4'd0, fsm_current_state};
                    cmd_state <= CS_IDLE;
                end
            end
        endcase

        // Auto-send emitted tokens
        if (fsm_emit_valid && uart_tx_ready && cmd_state == CS_IDLE) begin
            uart_tx_valid <= 1'b1;
            uart_tx_data <= fsm_emit_token[7:0];  // Send low byte
        end
    end
end

// ============================================================================
// RGB LED STATUS (Active Low)
// ============================================================================
// Red: Halted
// Green: Running (emit pulse)
// Blue: Ready

wire led_red, led_green, led_blue;

assign led_red   = fsm_halted;
assign led_green = fsm_emit_valid;
assign led_blue  = fsm_ready;

`ifdef SYNTHESIS
    // RGB LED driver primitive
    SB_RGBA_DRV #(
        .CURRENT_MODE("0b1"),       // Half current
        .RGB0_CURRENT("0b000001"),  // 4mA
        .RGB1_CURRENT("0b000001"),
        .RGB2_CURRENT("0b000001")
    ) rgb_drv_inst (
        .CURREN(1'b1),
        .RGBLEDEN(1'b1),
        .RGB0PWM(led_green),
        .RGB1PWM(led_blue),
        .RGB2PWM(led_red),
        .RGB0(led_green_n),
        .RGB1(led_blue_n),
        .RGB2(led_red_n)
    );
`else
    assign led_red_n   = ~led_red;
    assign led_green_n = ~led_green;
    assign led_blue_n  = ~led_blue;
`endif

// ============================================================================
// SPI FLASH (unused for now)
// ============================================================================
assign spi_cs_n = 1'b1;
assign spi_sck  = 1'b0;
assign spi_mosi = 1'b0;

// ============================================================================
// GPIO (directly accessible)
// ============================================================================
assign gpio = 8'bz;  // High-Z

endmodule

// ============================================================================
// UART TX MODULE
// ============================================================================
module uart_tx #(
    parameter CLK_DIV = 416
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] data,
    input  wire       valid,
    output reg        ready,
    output reg        tx
);

reg [15:0] counter;
reg [3:0]  bit_idx;
reg [9:0]  shift_reg;  // Start + 8 data + Stop
reg        busy;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        counter <= 16'd0;
        bit_idx <= 4'd0;
        shift_reg <= 10'h3FF;
        busy <= 1'b0;
        ready <= 1'b1;
        tx <= 1'b1;
    end else begin
        if (!busy) begin
            ready <= 1'b1;
            if (valid) begin
                shift_reg <= {1'b1, data, 1'b0};  // Stop, Data, Start
                busy <= 1'b1;
                ready <= 1'b0;
                counter <= 16'd0;
                bit_idx <= 4'd0;
            end
        end else begin
            if (counter < CLK_DIV - 1) begin
                counter <= counter + 16'd1;
            end else begin
                counter <= 16'd0;
                tx <= shift_reg[0];
                shift_reg <= {1'b1, shift_reg[9:1]};
                bit_idx <= bit_idx + 4'd1;
                if (bit_idx == 4'd9) begin
                    busy <= 1'b0;
                end
            end
        end
    end
end

endmodule

// ============================================================================
// UART RX MODULE
// ============================================================================
module uart_rx #(
    parameter CLK_DIV = 416
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,
    output reg  [7:0] data,
    output reg        valid,
    input  wire       ready
);

reg [15:0] counter;
reg [3:0]  bit_idx;
reg [7:0]  shift_reg;
reg        busy;
reg        rx_sync1, rx_sync2;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        counter <= 16'd0;
        bit_idx <= 4'd0;
        shift_reg <= 8'd0;
        busy <= 1'b0;
        valid <= 1'b0;
        data <= 8'd0;
        rx_sync1 <= 1'b1;
        rx_sync2 <= 1'b1;
    end else begin
        // Synchronize RX
        rx_sync1 <= rx;
        rx_sync2 <= rx_sync1;

        // Clear valid when data is read
        if (valid && ready) begin
            valid <= 1'b0;
        end

        if (!busy) begin
            // Wait for start bit
            if (!rx_sync2) begin
                busy <= 1'b1;
                counter <= CLK_DIV / 2;  // Sample at middle of bit
                bit_idx <= 4'd0;
            end
        end else begin
            if (counter < CLK_DIV - 1) begin
                counter <= counter + 16'd1;
            end else begin
                counter <= 16'd0;
                if (bit_idx == 4'd0) begin
                    // Verify start bit
                    if (rx_sync2) begin
                        busy <= 1'b0;  // False start
                    end else begin
                        bit_idx <= 4'd1;
                    end
                end else if (bit_idx <= 4'd8) begin
                    shift_reg <= {rx_sync2, shift_reg[7:1]};
                    bit_idx <= bit_idx + 4'd1;
                end else begin
                    // Stop bit
                    if (rx_sync2) begin
                        data <= shift_reg;
                        valid <= 1'b1;
                    end
                    busy <= 1'b0;
                end
            end
        end
    end
end

endmodule

`default_nettype wire
