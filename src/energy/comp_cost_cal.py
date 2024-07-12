from arithmetics import *

def tppe():

    sudo_acc = Accumulator('seudo_acc', 12)
    sudo_acc_area = sudo_acc.get_area()
    sudo_acc_dp = sudo_acc.get_dpower()
    sudo_acc_lp = sudo_acc.get_lpower()
    sudo_acc_e = sudo_acc.get_access_energy()
    
    correction_acc = Accumulator('corr_acc', 10)
    correction_acc_area = correction_acc.get_area()*4
    correction_acc_dp = correction_acc.get_dpower()*4
    correction_acc_lp = correction_acc.get_lpower()*4
    correction_acc_e = correction_acc.get_access_energy()*4

    #! SparTen paper Table 4
    fast_prefix_scale_power = 3.94
    fast_prefix_scale_area = 11.125

    dummy_acc = Accumulator('dummy', 24) 
    dummy_mul = Multiplier('dummy', 8)

    fast_prefix_area = (dummy_acc.get_area()+dummy_mul.get_area())*fast_prefix_scale_area
    fast_prefix_dp = (dummy_acc.get_dpower()+dummy_mul.get_dpower())*fast_prefix_scale_power
    fast_prefix_lp = (dummy_acc.get_lpower()+dummy_mul.get_lpower())*fast_prefix_scale_power
    fast_prefix_e = fast_prefix_dp*2.5


    laggy_prefix_adder = Adder('1/16', 3) #! small adder is 3-bit inside the laggy prefix
    laggy_prefix_buffer = Register('128', 128)

    laggy_prefix_area = laggy_prefix_adder.get_area()*16 + laggy_prefix_buffer.get_area()
    laggy_prefix_dp = laggy_prefix_adder.get_dpower()*16 + laggy_prefix_buffer.get_dpower()
    laggy_prefix_lp = laggy_prefix_adder.get_lpower()*16 + laggy_prefix_buffer.get_lpower()
    laggy_prefix_e = laggy_prefix_adder.get_access_energy()*16 + laggy_prefix_buffer.get_access_energy()

    fifo_mp = FIFO('8/16',7)
    fifo_mp_area = fifo_mp.get_area()/2
    fifo_mp_dp = fifo_mp.get_dpower()/2
    fifo_mp_lp = fifo_mp.get_lpower()/2
    fifo_mp_e = fifo_mp.get_access_energy()/2

    fifo_b = FIFO('8/16',8)
    fifo_b_area = fifo_b.get_area()/2
    fifo_b_dp = fifo_b.get_dpower()/2
    fifo_b_lp = fifo_b.get_lpower()/2
    fifo_b_e = fifo_b.get_access_energy()/2

    buffer_stream = ScratchPad('stream',128*8)
    buffer_stream_area = buffer_stream.get_area()
    buffer_stream_dp = buffer_stream.get_dpower()
    buffer_stream_lp = buffer_stream.get_lpower()
    buffer_stream_e = buffer_stream.get_access_energy()
    
    buffer_bma = Register('bma',128)
    buffer_bma_area = buffer_bma.get_area()
    buffer_bma_dp = buffer_bma.get_dpower()
    buffer_bma_lp = buffer_bma.get_lpower()
    buffer_bma_e = buffer_bma.get_access_energy()
    
    buffer_bmb = Register('bmb',128)
    buffer_bmb_area = buffer_bmb.get_area()
    buffer_bmb_dp = buffer_bmb.get_dpower()
    buffer_bmb_lp = buffer_bmb.get_lpower()
    buffer_bmb_e = buffer_bmb.get_access_energy()

    total_area = sudo_acc_area + correction_acc_area + fast_prefix_area + laggy_prefix_area + fifo_mp_area + fifo_b_area + buffer_stream_area + buffer_bma_area + buffer_bmb_area
    total_dp = sudo_acc_dp + correction_acc_dp + fast_prefix_dp + laggy_prefix_dp + fifo_mp_dp + fifo_b_dp + buffer_stream_dp + buffer_bma_dp + buffer_bmb_dp
    total_lp = sudo_acc_lp + correction_acc_lp + fast_prefix_lp + laggy_prefix_lp + fifo_mp_lp + fifo_b_lp + buffer_stream_lp + buffer_bma_lp + buffer_bmb_lp
    total_e = sudo_acc_e + correction_acc_e + fast_prefix_e + laggy_prefix_e + fifo_mp_e + fifo_b_e + buffer_stream_e + buffer_bma_e + buffer_bmb_e

    # print('area breakup:', [sudo_acc_area/total_area, correction_acc_area/total_area, fast_prefix_area/total_area, laggy_prefix_area/total_area, fifo_mp_area/total_area, fifo_b_area/total_area, buffer_stream_area/total_area, buffer_bma_area/total_area, buffer_bmb_area/total_area])
    # total_p = total_dp + total_lp
    # print(total_p)
    # print('power ratio:', [(sudo_acc_dp + sudo_acc_lp)/total_p, (correction_acc_dp + correction_acc_lp)/total_p, (fast_prefix_dp + fast_prefix_lp)/total_p, (laggy_prefix_dp + laggy_prefix_lp)/total_p, (fifo_mp_dp + fifo_b_dp + buffer_stream_dp + buffer_bma_dp + buffer_bmb_dp + fifo_mp_lp + fifo_b_lp + buffer_stream_lp + buffer_bma_lp + buffer_bmb_lp)/total_p])
    # print('power break:', [(sudo_acc_dp + sudo_acc_lp), (correction_acc_dp + correction_acc_lp), (fast_prefix_dp + fast_prefix_lp), (laggy_prefix_dp + laggy_prefix_lp), (fifo_mp_dp + fifo_b_dp + buffer_stream_dp + buffer_bma_dp + buffer_bmb_dp + fifo_mp_lp + fifo_b_lp + buffer_stream_lp + buffer_bma_lp + buffer_bmb_lp)])


    # print('dpower breakup:', [sudo_acc_dp/total_dp, correction_acc_dp/total_dp, fast_prefix_dp/total_dp, laggy_prefix_dp/total_dp, fifo_mp_dp/total_dp, fifo_b_dp/total_dp, buffer_stream_dp/total_dp, buffer_bma_dp/total_dp, buffer_bmb_dp/total_dp])
    # print('lpower breakup:', [sudo_acc_lp/total_lp, correction_acc_lp/total_lp, fast_prefix_lp/total_lp, laggy_prefix_lp/total_lp, fifo_mp_lp/total_lp, fifo_b_lp/total_lp, buffer_stream_lp/total_lp, buffer_bma_lp/total_lp, buffer_bmb_lp/total_lp])
    # print('e breakup:', [sudo_acc_e/total_e, correction_acc_e/total_e, fast_prefix_e/total_e, laggy_prefix_e/total_e, fifo_mp_e/total_e, fifo_b_e/total_e, buffer_stream_e/total_e, buffer_bma_e/total_e, buffer_bmb_e/total_e])

    # print(laggy_prefix_adder.get_area()*16/total_area)
    # print(laggy_prefix_adder.get_dpower()*16/total_dp)
    # print(laggy_prefix_adder.get_lpower()*16/total_lp)
    # print(laggy_prefix_adder.get_access_energy()*16/total_e)

    # print(laggy_prefix_buffer.get_area()/total_area)
    # print(laggy_prefix_buffer.get_dpower()/total_dp)
    # print(laggy_prefix_buffer.get_lpower()/total_lp)
    # print(laggy_prefix_buffer.get_access_energy()/total_e)
    print('TPPE data')
    print(f'Total area of {total_area} mm^3')
    print(f'Total dpower of {total_dp} mW')
    print(f'Total lpower of {total_lp} mW')
    print(f'Total energy of {total_e} nJ')

    return total_area, total_dp, total_lp, total_e

# tppe_area, tppe_dp, tppe_lp, tppe_e = tppe()
# print('TPPE data')
# print(f'Total area of {tppe_area} mm^3')
# print(f'Total dpower of {tppe_dp} mw')
# print(f'Total lpower of {tppe_lp} mW')
# print(f'Total energy of {tppe_e} nJ')


def sparten_pe():

    fast_prefix_scale_power = 3.94
    fast_prefix_scale_area = 11.125

    dummy_acc = Accumulator('dummy', 24) 
    dummy_mul = Multiplier('dummy', 8)

    #! 1 fast prefix
    fast_prefix_area = (dummy_acc.get_area()+dummy_mul.get_area())*fast_prefix_scale_area
    fast_prefix_dp = (dummy_acc.get_dpower()+dummy_mul.get_dpower())*fast_prefix_scale_power
    fast_prefix_lp = (dummy_acc.get_lpower()+dummy_mul.get_lpower())*fast_prefix_scale_power
    fast_prefix_e = fast_prefix_dp*2.5

    sudo_acc = Accumulator('seudo_acc', 12)
    sudo_acc_area = sudo_acc.get_area()
    sudo_acc_dp = sudo_acc.get_dpower()
    sudo_acc_lp = sudo_acc.get_lpower()
    sudo_acc_e = sudo_acc.get_access_energy()

    buffer_bma = Register('bma',128*4)
    buffer_bma_area = buffer_bma.get_area()
    buffer_bma_dp = buffer_bma.get_dpower()
    buffer_bma_lp = buffer_bma.get_lpower()
    buffer_bma_e = buffer_bma.get_access_energy()
    
    buffer_bmb = Register('bmb',128)
    buffer_bmb_area = buffer_bmb.get_area()
    buffer_bmb_dp = buffer_bmb.get_dpower()
    buffer_bmb_lp = buffer_bmb.get_lpower()
    buffer_bmb_e = buffer_bmb.get_access_energy()

    buffer_stream = ScratchPad('stream',128*8)
    buffer_stream_area = buffer_stream.get_area()
    buffer_stream_dp = buffer_stream.get_dpower()
    buffer_stream_lp = buffer_stream.get_lpower()
    buffer_stream_e = buffer_stream.get_access_energy()

    total_area = sudo_acc_area + fast_prefix_area + buffer_bma_area + buffer_bmb_area + buffer_stream_area
    total_dp = sudo_acc_dp + fast_prefix_dp + buffer_bma_dp + buffer_bmb_dp + buffer_stream_dp
    total_lp = sudo_acc_lp + fast_prefix_lp + buffer_bma_lp + buffer_bmb_lp + buffer_stream_lp

    #! -> The single compute path = (bma + bmb + buffer_stream) - > fast_prefix -> acc
    total_e = sudo_acc_e + fast_prefix_e + buffer_bma_e + buffer_bmb_e + buffer_stream_e
    
    print('SparTen data')
    print(f'Total area of {total_area} mm^3')
    print(f'Total dpower of {total_dp} mW')
    print(f'Total lpower of {total_lp} mW')
    print(f'Total energy of {total_e} nJ')

    return total_area, total_dp, total_lp, total_e

# sparten_pe_area, sparten_pe_dp, sparten_pe_lp, sparten_pe_e = sparten_pe()
# print('SparTen data')
# print(f'Total area of {sparten_pe_area} mm^3')
# print(f'Total dpower of {sparten_pe_dp} mw')
# print(f'Total lpower of {sparten_pe_lp} mW')
# print(f'Total energy of {sparten_pe_e} nJ')



def sparten_ann_pe():

    fast_prefix_scale_power = 3.94
    fast_prefix_scale_area = 11.125

    dummy_acc = Accumulator('dummy', 24) 
    dummy_mul = Multiplier('dummy', 8)

    #! 2 fast prefix
    fast_prefix_area = (dummy_acc.get_area()+dummy_mul.get_area())*fast_prefix_scale_area*2
    fast_prefix_dp = (dummy_acc.get_dpower()+dummy_mul.get_dpower())*fast_prefix_scale_power*2
    fast_prefix_lp = (dummy_acc.get_lpower()+dummy_mul.get_lpower())*fast_prefix_scale_power*2
    fast_prefix_e = fast_prefix_dp*2.5

    sudo_acc = Accumulator('seudo_acc', 24)
    sudo_acc_area = sudo_acc.get_area()
    sudo_acc_dp = sudo_acc.get_dpower()
    sudo_acc_lp = sudo_acc.get_lpower()
    sudo_acc_e = sudo_acc.get_access_energy()

    sudo_mul = Multiplier('seudo_acc', 8)
    sudo_mul_area = sudo_mul.get_area()
    sudo_mul_dp = sudo_mul.get_dpower()
    sudo_mul_lp = sudo_mul.get_lpower()
    sudo_mul_e = sudo_mul.get_access_energy()

    buffer_bma = Register('bma',128)
    buffer_bma_area = buffer_bma.get_area()
    buffer_bma_dp = buffer_bma.get_dpower()
    buffer_bma_lp = buffer_bma.get_lpower()
    buffer_bma_e = buffer_bma.get_access_energy()
    
    buffer_bmb = Register('bmb',128)
    buffer_bmb_area = buffer_bmb.get_area()
    buffer_bmb_dp = buffer_bmb.get_dpower()
    buffer_bmb_lp = buffer_bmb.get_lpower()
    buffer_bmb_e = buffer_bmb.get_access_energy()

    buffer_stream = ScratchPad('stream',128*8)
    buffer_stream_area = buffer_stream.get_area()
    buffer_stream_dp = buffer_stream.get_dpower()
    buffer_stream_lp = buffer_stream.get_lpower()
    buffer_stream_e = buffer_stream.get_access_energy()

    total_area = sudo_acc_area + sudo_mul_area + fast_prefix_area + buffer_bma_area + buffer_bmb_area + buffer_stream_area
    total_dp = sudo_acc_dp + fast_prefix_dp + sudo_mul_dp + buffer_bma_dp + buffer_bmb_dp + buffer_stream_dp
    total_lp = sudo_acc_lp + sudo_mul_dp + fast_prefix_lp + buffer_bma_lp + buffer_bmb_lp + buffer_stream_lp

    #! -> The single compute path = (bma + bmb + buffer_stream) - > fast_prefix -> acc
    total_e = sudo_acc_e + sudo_mul_e + fast_prefix_e + buffer_bma_e + buffer_bmb_e + buffer_stream_e
    print('SparTen ANN data')
    print(f'Total area of {total_area} mm^3')
    print(f'Total dpower of {total_dp} mW')
    print(f'Total lpower of {total_lp} mW')
    print(f'Total energy of {total_e} nJ')

    return total_area, total_dp, total_lp, total_e

# sparten_ann_pe_area, sparten_ann_pe_dp, sparten_ann_pe_lp, sparten_ann_pe_e = sparten_ann_pe()
# print('SparTen ann data')
# print(f'Total area of {sparten_ann_pe_area} mm^3')
# print(f'Total dpower of {sparten_ann_pe_dp} mw')
# print(f'Total lpower of {sparten_ann_pe_lp} mW')
# print(f'Total energy of {sparten_ann_pe_e} nJ')



def gospa_pe():

    sudo_acc = Accumulator('seudo_acc', 12)
    sudo_acc_area = sudo_acc.get_area()*256*4
    sudo_acc_dp = sudo_acc.get_dpower()*256*4
    sudo_acc_lp = sudo_acc.get_lpower()*256*4
    # sudo_acc_e = sudo_acc.get_access_energy()*256*4

    buffer_bma = Register('bma',8)
    buffer_bma_area = buffer_bma.get_area()*2
    buffer_bma_dp = buffer_bma.get_dpower()*2
    buffer_bma_lp = buffer_bma.get_lpower()*2
    # buffer_bma_e = buffer_bma.get_access_energy()*2

    fifo_mp = FIFO('64/16',14)
    fifo_mp_area = fifo_mp.get_area()*4
    fifo_mp_dp = fifo_mp.get_dpower()*4
    fifo_mp_lp = fifo_mp.get_lpower()*4
    # fifo_mp_e = fifo_mp.get_access_energy()*4


    total_area = sudo_acc_area + fifo_mp_area + buffer_bma_area
    total_dp = sudo_acc_dp + fifo_mp_dp + buffer_bma_dp
    total_lp = sudo_acc_lp + fifo_mp_lp + buffer_bma_lp

    #! -> The single compute path = (FIFO + buffer) - > Acc
    total_e = sudo_acc.get_access_energy()*4 + fifo_mp.get_access_energy() + buffer_bma.get_access_energy()
    print('GoSPA data')
    print(f'Total area of {total_area} mm^3')
    print(f'Total dpower of {total_dp} mW')
    print(f'Total lpower of {total_lp} mW')
    print(f'Total energy of {total_e} nJ')

    return total_area, total_dp, total_lp, total_e

# gospa_pe_area, gospa_pe_dp, gospa_pe_lp, gospa_pe_e = gospa_pe()
# print('GoSPA data')
# print(f'Total area of {gospa_pe_area} mm^3')
# print(f'Total dpower of {gospa_pe_dp} mw')
# print(f'Total lpower of {gospa_pe_lp} mW')
# print(f'Total energy of {gospa_pe_e} nJ')


def gamma_pe():

    acc = Accumulator('gamma_acc', 24)
    acc_area = acc.get_area()
    acc_dp = acc.get_dpower()
    acc_lp = acc.get_lpower()

    buffer_x = Register('buf_x',64)
    buffer_x_area = buffer_x.get_area()
    buffer_x_dp = buffer_x.get_dpower()
    buffer_x_lp = buffer_x.get_lpower()

    buffer_w = Register('buf_w',64*8)
    buffer_w_area = buffer_w.get_area()
    buffer_w_dp = buffer_w.get_dpower()
    buffer_w_lp = buffer_w.get_lpower()

    buffer_wcsr = Register('buf_w',64*6)
    buffer_wcsr_area = buffer_wcsr.get_area()
    buffer_wcsr_dp = buffer_wcsr.get_dpower()
    buffer_wcsr_lp = buffer_wcsr.get_lpower()

    #! The merger is rougly 55% of 64-bit Multiplier's area
    dummy_mul = Multiplier('dummy', 64)
    merger_area = dummy_mul.get_area()*0.55
    merger_dp = dummy_mul.get_dpower()*0.55
    merger_lp = dummy_mul.get_lpower()*0.55


    total_area = acc_area + buffer_x_area + buffer_w_area + buffer_wcsr_area + merger_area
    total_dp = acc_dp + buffer_x_dp + buffer_w_dp + buffer_wcsr_dp + merger_dp
    total_lp = acc_lp + buffer_x_lp + buffer_w_lp + buffer_wcsr_lp + merger_lp

    #! -> The single compute path = wcsr->merger + (bufferw + bufferx)-> Acc
    total_e = acc.get_access_energy() + buffer_x.get_access_energy() + buffer_w.get_access_energy() + buffer_wcsr.get_access_energy() + dummy_mul.get_access_energy()*0.55

    print('Gamma data')
    print(f'Total area of {total_area} mm^3')
    print(f'Total dpower of {total_dp} mW')
    print(f'Total lpower of {total_lp} mW')
    print(f'Total energy of {total_e} nJ')

    return total_area, total_dp, total_lp, total_e

def gamma_ann_pe():

    acc = Accumulator('gamma_acc', 24)
    acc_area = acc.get_area()
    acc_dp = acc.get_dpower()
    acc_lp = acc.get_lpower()

    buffer_x = Register('buf_x',64)
    buffer_x_area = buffer_x.get_area()
    buffer_x_dp = buffer_x.get_dpower()
    buffer_x_lp = buffer_x.get_lpower()

    buffer_w = Register('buf_w',64*8)
    buffer_w_area = buffer_w.get_area()
    buffer_w_dp = buffer_w.get_dpower()
    buffer_w_lp = buffer_w.get_lpower()

    buffer_wcsr = Register('buf_w',64*6)
    buffer_wcsr_area = buffer_wcsr.get_area()
    buffer_wcsr_dp = buffer_wcsr.get_dpower()
    buffer_wcsr_lp = buffer_wcsr.get_lpower()

    #! The merger is rougly 55% of 64-bit Multiplier's area
    dummy_mul = Multiplier('dummy', 64)
    merger_area = dummy_mul.get_area()*0.55
    merger_dp = dummy_mul.get_dpower()*0.55
    merger_lp = dummy_mul.get_lpower()*0.55

    mul = Multiplier('dummy', 8)
    mul_area = mul.get_area()
    mul_dp = mul.get_dpower()
    mul_lp = mul.get_lpower()
    mul_e = mul.get_access_energy()


    total_area = acc_area + buffer_x_area + buffer_w_area + buffer_wcsr_area + merger_area + mul_area
    total_dp = acc_dp + buffer_x_dp + buffer_w_dp + buffer_wcsr_dp + merger_dp + mul_dp
    total_lp = acc_lp + buffer_x_lp + buffer_w_lp + buffer_wcsr_lp + merger_lp + mul_lp

    #! -> The single compute path = wcsr->merger + (bufferw + bufferx)-> Acc
    total_e = acc.get_access_energy() + buffer_x.get_access_energy() + buffer_w.get_access_energy() + buffer_wcsr.get_access_energy() + dummy_mul.get_access_energy()*0.55 + mul.get_access_energy()

    print('Gamma ANN data')
    print(f'Total area of {total_area} mm^3')
    print(f'Total dpower of {total_dp} mW')
    print(f'Total lpower of {total_lp} mW')
    print(f'Total energy of {total_e} nJ')

    return total_area, total_dp, total_lp, total_e

# gospa_pe_area, gospa_pe_dp, gospa_pe_lp, gospa_pe_e = gospa_pe()


def ptb_pe():

    #! total 4 8-bit regs
    b_reg_8 = Register('8-b',8)
    b_reg_8_dp = b_reg_8.get_dpower()*4
    b_reg_8_lp = b_reg_8.get_lpower()*4

    sudo_acc = Accumulator('seudo_acc', 12)
    sudo_acc_dp = sudo_acc.get_dpower()
    sudo_acc_lp = sudo_acc.get_lpower()

    total_dp = b_reg_8_dp + sudo_acc_dp
    total_lp = b_reg_8_lp + sudo_acc_lp
    total_e = sudo_acc.get_access_energy() + b_reg_8.get_access_energy()*2

    print('PTB data')
    # ptb_pe_dp, ptb_pe_lp, ptb_pe_e = ptb_pe()
    print(f'Total dpower of {total_dp} mW')
    print(f'Total lpower of {total_lp} mW')
    print(f'Total energy of {total_e} nJ')

    return total_dp, total_lp, total_e

def stellar_pe():

    b_reg_1 = Register('12x6-b',72)
    b_reg_1_dp = b_reg_1.get_dpower()
    b_reg_1_lp = b_reg_1.get_lpower()

    b_reg_2 = Register('12x12-b',144)
    b_reg_2_dp = b_reg_2.get_dpower()
    b_reg_2_lp = b_reg_2.get_lpower()

    b_reg_3 = Register('24x12-b',288)
    b_reg_3_dp = b_reg_3.get_dpower()
    b_reg_3_lp = b_reg_3.get_lpower()

    sudo_acc = Accumulator('seudo_acc', 12)
    sudo_acc_dp = sudo_acc.get_dpower()*4
    sudo_acc_lp = sudo_acc.get_lpower()*4

    total_dp = b_reg_1_dp + b_reg_2_dp + b_reg_3_dp + sudo_acc_dp
    total_lp = b_reg_1_lp + b_reg_2_lp + b_reg_3_lp + sudo_acc_lp
    total_e = sudo_acc.get_access_energy() + b_reg_1.get_access_energy() + b_reg_2.get_access_energy() + b_reg_3.get_access_energy()

    print('Stellar data')
    # ptb_pe_dp, ptb_pe_lp, ptb_pe_e = ptb_pe()
    print(f'Total dpower of {total_dp} mW')
    print(f'Total lpower of {total_lp} mW')
    print(f'Total energy of {total_e} nJ')

    return total_dp, total_lp, total_e

# gamma_pe()
# stellar_pe()
gamma_ann_pe()