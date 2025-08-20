`timescale 1ns / 1ps
module accelerator_top #(
    parameter ADDR_WIDTH = 7,  
    parameter DATA_WIDTH = 32
)(
    // AXI4-Lite slave interface
    input  logic                      s_axi_aclk,
    input  logic                      s_axi_aresetn,

    // Write address channel
    input  logic [ADDR_WIDTH-1:0]     s_axi_awaddr,
    input  logic                      s_axi_awvalid,
    output logic                      s_axi_awready,

    // Write data channel
    input  logic [DATA_WIDTH-1:0]     s_axi_wdata,
    input  logic [(DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  logic                      s_axi_wvalid,
    output logic                      s_axi_wready,

    // Write response channel
    output logic [1:0]                s_axi_bresp,
    output logic                      s_axi_bvalid,
    input  logic                      s_axi_bready,

    // Read address channel
    input  logic [ADDR_WIDTH-1:0]     s_axi_araddr,
    input  logic                      s_axi_arvalid,
    output logic                      s_axi_arready,

    // Read data channel
    output logic [DATA_WIDTH-1:0]     s_axi_rdata,
    output logic [1:0]                s_axi_rresp,
    output logic                      s_axi_rvalid,
    input  logic                      s_axi_rready
);

    // Local register offsets (match C example)
    localparam OFF_DATA_BASE   = 8'h00; // 0x00
    localparam OFF_WEI_BASE    = 8'h20; // 0x20
    localparam OFF_BIAS        = 8'h40; // 0x40
    localparam OFF_CTRL        = 8'h44; // 0x44
    localparam OFF_STATUS      = 8'h48; // 0x48
    localparam OFF_RESULT      = 8'h4C; // 0x4C

    // Simple AXI write address / data handshake
    logic [ADDR_WIDTH-1:0] write_addr;
    logic                  write_addr_valid;
    logic                  write_data_ready;

    // capture write address
    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            s_axi_awready <= 1'b0;
            write_addr_valid <= 1'b0;
            write_addr <= '0;
        end else begin
            // awready asserted when not holding an address
            if (!write_addr_valid && !s_axi_awready) s_axi_awready <= 1'b1;
            else s_axi_awready <= 1'b0;

            if (s_axi_awvalid && s_axi_awready) begin
                write_addr <= s_axi_awaddr;
                write_addr_valid <= 1'b1;
            end

            // clear once write completes (handled on wvalid)
            if (write_data_ready) write_addr_valid <= 1'b0;
        end
    end

    // write data handshake
    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            s_axi_wready <= 1'b0;
            write_data_ready <= 1'b0;
        end else begin
            // accept write data when we have an address
            s_axi_wready <= write_addr_valid;
            if (s_axi_wvalid && s_axi_wready) begin
                write_data_ready <= 1'b1;
            end else begin
                write_data_ready <= 1'b0;
            end
        end
    end

    // write response
    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            s_axi_bvalid <= 1'b0;
            s_axi_bresp  <= 2'b00;
        end else begin
            if (write_data_ready) begin
                s_axi_bvalid <= 1'b1;
                s_axi_bresp  <= 2'b00; // OKAY
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end
        end
    end

    // Read address handshake
    logic [ADDR_WIDTH-1:0] read_addr;
    logic                  read_addr_valid;

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            s_axi_arready <= 1'b0;
            read_addr_valid <= 1'b0;
            read_addr <= '0;
        end else begin
            if (!read_addr_valid && !s_axi_arready) s_axi_arready <= 1'b1;
            else s_axi_arready <= 1'b0;

            if (s_axi_arvalid && s_axi_arready) begin
                read_addr <= s_axi_araddr;
                read_addr_valid <= 1'b1;
            end

            // cleared when rvalid accepted
            if (s_axi_rvalid && s_axi_rready) read_addr_valid <= 1'b0;
        end
    end

    // Register bank
    logic [31:0] data_regs [0:7];
    logic [31:0] weight_regs [0:7];
    logic [31:0] bias_reg;
    logic        ctrl_start;

    // Control signals for the accelerator core
    logic        core_done;
    logic        core_busy;
    logic [31:0] core_result;

    // default resets
    integer i;
    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            for (i=0;i<8;i++) begin
                data_regs[i] <= 32'h0;
                weight_regs[i] <= 32'h0;
            end
            bias_reg <= 32'h0;
            ctrl_start <= 1'b0;
        end else begin
            // perform register write when both address and data present
            if (write_data_ready) begin
                // decode by address (byte address)
                unique case (write_addr & 32'hFF)  // mask to low 8 bits
                    // data regs 0x00..0x1C : stride 4
                    8'h00,8'h04,8'h08,8'h0C,8'h10,8'h14,8'h18,8'h1C : begin
                        int idx = (write_addr - OFF_DATA_BASE) >> 2;
                        if (idx >= 0 && idx < 8) begin
                            data_regs[idx] <= s_axi_wdata;
                        end
                    end
                    // weight regs 0x20..0x3C
                    8'h20,8'h24,8'h28,8'h2C,8'h30,8'h34,8'h38,8'h3C : begin
                        int idx = (write_addr - OFF_WEI_BASE) >> 2;
                        if (idx >= 0 && idx < 8) begin
                            weight_regs[idx] <= s_axi_wdata;
                        end
                    end
                    8'h40: begin
                        bias_reg <= s_axi_wdata;
                    end
                    8'h44: begin
                        // write START bit
                        if (s_axi_wdata[0]) begin
                            // set start pulse (core will sample)
                            ctrl_start <= 1'b1;
                        end
                    end
                    default: begin
                        // ignore other writes
                    end
                endcase
            end else begin
                // clear start after one cycle so it's just a pulse
                ctrl_start <= 1'b0;
            end
        end
    end

    // Read data path: form rdata when read_addr_valid asserted
    always_comb begin
        s_axi_rdata = 32'h0;
        s_axi_rresp = 2'b00; // OKAY
        if (read_addr_valid) begin
            unique case (read_addr & 32'hFF)
                // map data regs
                8'h00: s_axi_rdata = data_regs[0];
                8'h04: s_axi_rdata = data_regs[1];
                8'h08: s_axi_rdata = data_regs[2];
                8'h0C: s_axi_rdata = data_regs[3];
                8'h10: s_axi_rdata = data_regs[4];
                8'h14: s_axi_rdata = data_regs[5];
                8'h18: s_axi_rdata = data_regs[6];
                8'h1C: s_axi_rdata = data_regs[7];
                // weights
                8'h20: s_axi_rdata = weight_regs[0];
                8'h24: s_axi_rdata = weight_regs[1];
                8'h28: s_axi_rdata = weight_regs[2];
                8'h2C: s_axi_rdata = weight_regs[3];
                8'h30: s_axi_rdata = weight_regs[4];
                8'h34: s_axi_rdata = weight_regs[5];
                8'h38: s_axi_rdata = weight_regs[6];
                8'h3C: s_axi_rdata = weight_regs[7];
                8'h40: s_axi_rdata = bias_reg;
                8'h44: s_axi_rdata = 32'h0; // CTRL reads 0
                8'h48: s_axi_rdata = {30'b0, core_busy, core_done};
                8'h4C: s_axi_rdata = core_result;
                default: s_axi_rdata = 32'h0;
            endcase
        end
    end

    // assert rvalid when read_addr_valid captured, and deassert after accept
    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            s_axi_rvalid <= 1'b0;
        end else begin
            if (read_addr_valid && !s_axi_rvalid) begin
                s_axi_rvalid <= 1'b1;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

    logic [255:0] data_vec;
    logic [255:0] weight_vec;
    // pack: data_regs[0] holds data[0] (low 16) and data[1] (high 16)
    genvar gi;
    generate
      for (gi = 0; gi < 8; gi = gi + 1) begin : PACK
        assign data_vec[(gi*32) +: 32] = data_regs[gi];
        assign weight_vec[(gi*32) +: 32] = weight_regs[gi];
      end
    endgenerate

    // core control signals
    logic core_start;
    logic core_rst_n = s_axi_aresetn; // pass reset
    // pulse core_start when ctrl_start asserted and core not busy
    always_ff @(posedge s_axi_aclk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            core_start <= 1'b0;
        end else begin
            if (ctrl_start && !core_busy) core_start <= 1'b1;
            else core_start <= 1'b0;
        end
    end

    accelerator_core core_i (
        .clk      (s_axi_aclk),
        .rst_n    (core_rst_n),
        .start    (core_start),
        .data_in  (data_vec),
        .wei_in   (weight_vec),
        .bias_in  (bias_reg),
        .busy     (core_busy),
        .done     (core_done),
        .result   (core_result) 
    );

endmodule
