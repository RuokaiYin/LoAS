import torch
import time
import math
import torch.nn.functional as F
from tqdm import tqdm

#######                                               #######
####### Local Helper Functions for PE tile Generation #######
#######                                               #######
def _check_boundary_helper(avai_resources, dim_size):
    if avai_resources > dim_size:
        return dim_size
    else:
        return avai_resources

def _extra_fetch_dim2_helper(avai_resources, dim0_size, dim1_size):
    if avai_resources > dim0_size:
        if avai_resources//dim0_size > 1:
            return dim0_size, (avai_resources//dim0_size)
        else:
            return dim0_size, 1
    else:
        return avai_resources, 1

def _extra_fetch_dim3_helper(avai_resources, dim0_size, dim1_size, dim2_size):
    if avai_resources > dim0_size:
        if avai_resources > dim0_size*dim1_size:
            if avai_resources//(dim0_size*dim1_size) > 1:
                return dim0_size, dim1_size, avai_resources//(dim0_size*dim1_size)
            else:
                return dim0_size, dim1_size, 1
        else:
            dim0, dim1 = _extra_fetch_dim2_helper(avai_resources, dim0_size, dim1_size)
            return dim0, dim1, 1
    else:
        return avai_resources, 1, 1



#######                                                    #######
####### PE Level Tiling Shape Generation for Weight Matrix #######
#######                                                    #######
def _Tile_shape_Anl_W(pe_info, pe_dataflow, w_size):
    
    #? This means inter-PE parallel is not allowed.
    if pe_dataflow['w']['inter_pe'] == 0:

        #? First case: only one dimension is absorbed. 
        #? Meaning extra local buffers will not be used beyond the selected dim.
        if len(pe_dataflow['w']['intra_pe']) == 1:
            if pe_dataflow['w']['intra_pe'] == 'k':
                w_tile_r = _check_boundary_helper(pe_info['num_w'], w_size(0))
                w_tile_c = 1
            elif pe_dataflow['w']['intra_pe'] == 'n':
                w_tile_c = _check_boundary_helper(pe_info['num_w'], w_size(1))
                w_tile_r = 1
            else:
                assert "Wrong weight intra-pe dimensions other than w[k,n] is provided."
        
        #? Second case: two dims are allowed to be absorbed.
        #? Meaning if we have enough local buffers, we can fetch more than one primary dims 
        #? (extra # of fetch needs to be in integer.)
        elif len(pe_dataflow['w']['intra_pe']) == 2:
            if pe_dataflow['w']['intra_pe'][0] == 'k':
                w_tile_r, w_tile_c = _extra_fetch_dim2_helper(pe_info['num_w'], w_size(0), w_size(1))
            elif pe_dataflow['w']['intra_pe'][0] == 'n':
                w_tile_c, w_tile_r = _extra_fetch_dim2_helper(pe_info['num_w'], w_size(1), w_size(0))
            else:
                assert "Wrong weight intra-pe dimensions other than w[k,n] is provided."
        else:
            assert "wrong # of weight dimensions, more than 2 or less than 1."

    #? This means only inter-PE parallel is allowed.
    elif pe_dataflow['w']['intra_pe'] == 0:

        #? First case: only one dimension is PE expanding on.
        if len(pe_dataflow['w']['inter_pe']) == 1:
            if pe_dataflow['w']['inter_pe'] == 'k':
                w_tile_r = _check_boundary_helper(pe_info['num_pe'], w_size(0))
                w_tile_c = 1
            elif pe_dataflow['w']['inter_pe'] == 'n':
                w_tile_c = _check_boundary_helper(pe_info['num_pe'], w_size(1))
                w_tile_r = 1
            else:
                assert "Wrong weight intra-pe dimensions other than w[k,n] is provided."
        elif len(pe_dataflow['w']['inter_pe']) == 2:
            if pe_dataflow['w']['inter_pe'][0] == 'k':
                w_tile_r, w_tile_c = _extra_fetch_dim2_helper(pe_info['num_pe'], w_size(0), w_size(1))
            elif pe_dataflow['w']['inter_pe'][0] == 'n':
                w_tile_c, w_tile_r = _extra_fetch_dim2_helper(pe_info['num_pe'], w_size(1), w_size(0))
            else:
                assert "Wrong weight intra-pe dimensions other than w[k,n] is provided."
        else:
            assert "wrong # of weight dimensions, more than 2 or less than 1."
    
    #? A common case, allowing parallel PEs to receive different datas at same time.
    elif pe_dataflow['w']['intra_pe'] != 0 and pe_dataflow['w']['inter_pe'] != 0:

        if len(pe_dataflow['w']['inter_pe']) == 1 and len(pe_dataflow['w']['intra_pe']) == 1:
            if pe_dataflow['w']['inter_pe'] == 'n':
                if pe_dataflow['w']['intra_pe'] == 'k':
                    w_tile_r = _check_boundary_helper(pe_info['num_w'], w_size(0))
                    w_tile_c = _check_boundary_helper(pe_info['num_pe'], w_size(1))
                else:
                    assert "Wrong weight dimensions other than w[k,n] is provided."
            elif pe_dataflow['w']['inter_pe'] == 'k':
                if pe_dataflow['w']['intra_pe'] == 'n':
                    w_tile_c = _check_boundary_helper(pe_info['num_w'], w_size(1))
                    w_tile_r = _check_boundary_helper(pe_info['num_pe'], w_size(0))
                else:
                    assert "Wrong weight dimensions other than w[k,n] is provided."
            else:
                assert "Wrong weight dimensions other than w[k,n] is provided."
        else:
            assert "TODO:, support multi dimensions at inter and intra level."
    
    else:
        assert "TODO: support both inter and intra to be disabled. However, this should be very trivial and uncommon case."

    return (w_tile_r, w_tile_c)




#######                                                   #######
####### PE Level Tiling Shape Generation for INPUT Matrix #######
#######                                                   #######
def _Tile_shape_Anl_Inp(pe_info, pe_dataflow, inp_size):
    #! TODO: seems there's a easier way to combine the inter and intra parts.
    #! By using different keywords, making the codes easier to read.
    
    
     #? This means inter-PE parallel is not allowed. All PEs share the same input data.
    if pe_dataflow['x']['inter_pe'] == 0: 
        
        #? Following three cases only allow 1 dimension in intra-pe, meaning the filling of local buffers stop 
        #? once the selected dimension is consumed. Other dimension will always be 1 in the tile.
        if len(pe_dataflow['x']['intra_pe']) == 1:
            if pe_dataflow['x']['intra_pe'] == 't':
                inp_tile_t = _check_boundary_helper(pe_info['num_inp'], inp_size(0))
                inp_tile_c = 1
                inp_tile_r = 1
            elif pe_dataflow['x']['intra_pe'] == 'm':
                inp_tile_r = _check_boundary_helper(pe_info['num_inp'], inp_size(1))
                inp_tile_c = 1
                inp_tile_t = 1
            elif pe_dataflow['x']['intra_pe'] == 'k':
                inp_tile_c = _check_boundary_helper(pe_info['num_inp'], inp_size(2))
                inp_tile_r = 1
                inp_tile_t = 1
            else:
                assert "Wrong input intra dimensions other than [t,m,k]"
    #? Following cases allow 2 dimensions in intra_pe.
        elif len(pe_dataflow['x']['intra_pe']) == 2:
            #? Check the priortized absorbing dimension.
            if pe_dataflow['x']['intra_pe'][0] == 't':
                if pe_dataflow['x']['intra_pe'][1] == 'm':
                    inp_tile_t, inp_tile_r = _extra_fetch_dim2_helper(pe_info['num_inp'], inp_size(0), inp_size(1))
                    inp_tile_c = 1
                elif pe_dataflow['x']['intra_pe'][1] == 'k':
                    inp_tile_t, inp_tile_c = _extra_fetch_dim2_helper(pe_info['num_inp'], inp_size(0), inp_size(2))
                    inp_tile_r = 1
                else:
                    assert "wrong dimensions other than [t,m,k]"
            elif pe_dataflow['x']['intra_pe'][0] == 'm':
                if pe_dataflow['x']['intra_pe'][1] == 't':
                    inp_tile_r, inp_tile_t = _extra_fetch_dim2_helper(pe_info['num_inp'], inp_size(1), inp_size(0))
                    inp_tile_c = 1
                elif pe_dataflow['x']['intra_pe'][1] == 'k':
                    inp_tile_r, inp_tile_c = _extra_fetch_dim2_helper(pe_info['num_inp'], inp_size(1), inp_size(2))
                    inp_tile_t = 1
                else:
                    assert "wrong dimensions other than [t,m,k]"
            elif pe_dataflow['x']['intra_pe'][0] == 'k':
                if pe_dataflow['x']['intra_pe'][1] == 't':
                    inp_tile_c, inp_tile_t = _extra_fetch_dim2_helper(pe_info['num_inp'], inp_size(2), inp_size(0))
                    inp_tile_r = 1
                elif pe_dataflow['x']['intra_pe'][1] == 'm':
                    inp_tile_c, inp_tile_r = _extra_fetch_dim2_helper(pe_info['num_inp'], inp_size(2), inp_size(1))
                    inp_tile_t = 1
                else:
                    assert "wrong dimensions other than [t,m,k]"
            else:
                assert "wrong dimensions other than [t,m,k]"
        
        #? Following cases allow 3 dimensions in intra_pe.
        elif len(pe_dataflow['x']['intra_pe']) == 3:
            if pe_dataflow['x']['intra_pe'][0]=='t':
                if pe_dataflow['x']['intra_pe'][1]=='m':
                    inp_tile_t, inp_tile_r, inp_tile_c = _extra_fetch_dim3_helper(pe_info['num_inp'], inp_size(0), inp_size(1), inp_size(2))
                elif pe_dataflow['x']['intra_pe'][1]=='k':
                    inp_tile_t, inp_tile_c, inp_tile_r = _extra_fetch_dim3_helper(pe_info['num_inp'], inp_size(0), inp_size(2), inp_size(1))
                else:
                    assert "wrong dimensions other than [t,m,k]"
            elif pe_dataflow['x']['intra_pe'][0]=='m':
                if pe_dataflow['x']['intra_pe'][1]=='t':
                    inp_tile_r, inp_tile_t, inp_tile_c = _extra_fetch_dim3_helper(pe_info['num_inp'], inp_size(1), inp_size(0), inp_size(2))
                elif pe_dataflow['x']['intra_pe'][1]=='k':
                    inp_tile_r, inp_tile_c, inp_tile_t = _extra_fetch_dim3_helper(pe_info['num_inp'], inp_size(1), inp_size(2), inp_size(0))
                else:
                    assert "wrong dimensions other than [t,m,k]"
            elif pe_dataflow['x']['intra_pe'][0]=='k':
                if pe_dataflow['x']['intra_pe'][1]=='t':
                    inp_tile_c, inp_tile_t, inp_tile_r = _extra_fetch_dim3_helper(pe_info['num_inp'], inp_size(2), inp_size(0), inp_size(1))
                elif pe_dataflow['x']['intra_pe'][1]=='m':
                    inp_tile_c, inp_tile_r, inp_tile_t = _extra_fetch_dim3_helper(pe_info['num_inp'], inp_size(2), inp_size(1), inp_size(0))
                else:
                    assert "wrong dimensions other than [t,m,k]"
        else:
            assert "wrong # of input dimensions, more than 3 or less than 1."
    
    #? Only the parallel 
    #! TODO: to implement, where only the parallel is allowed. Might be applicable to systolic array.
    elif pe_dataflow['x']['intra_pe'] == 0:
        if len(pe_dataflow['x']['inter_pe']) == 1:
            if pe_dataflow['x']['inter_pe'] == 't':
                inp_tile_t = _check_boundary_helper(pe_info['num_pe'], inp_size(0))
                inp_tile_c = 1
                inp_tile_r = 1
            elif pe_dataflow['x']['inter_pe'] == 'm':
                inp_tile_r = _check_boundary_helper(pe_info['num_inp'], inp_size(1))
                inp_tile_c = 1
                inp_tile_t = 1
            elif pe_dataflow['x']['inter_pe'] == 'k':
                inp_tile_c = _check_boundary_helper(pe_info['num_inp'], inp_size(2))
                inp_tile_r = 1
                inp_tile_t = 1
            else:
                assert "wrong dimensions other than [t,m,k]"

        #? Following cases allow 2 dimensions in inter_pe.
        elif len(pe_dataflow['x']['inter_pe']) == 2:
            #? Check the priortized absorbing dimension.
            if pe_dataflow['x']['inter_pe'][0] == 't':
                if pe_dataflow['x']['inter_pe'][1] == 'm':
                    inp_tile_t, inp_tile_r = _extra_fetch_dim2_helper(pe_info['num_pe'], inp_size(0), inp_size(1))
                    inp_tile_c = 1
                elif pe_dataflow['x']['inter_pe'][1] == 'k':
                    inp_tile_t, inp_tile_c = _extra_fetch_dim2_helper(pe_info['num_pe'], inp_size(0), inp_size(2))
                    inp_tile_r = 1
                else:
                    assert "wrong dimensions other than [t,m,k]"

            elif pe_dataflow['x']['inter_pe'][0] == 'm':
                if pe_dataflow['x']['inter_pe'][1] == 't':
                    inp_tile_r, inp_tile_t = _extra_fetch_dim2_helper(pe_info['num_pe'], inp_size(1), inp_size(0))
                    inp_tile_c = 1
                elif pe_dataflow['x']['inter_pe'][1] == 'k':
                    inp_tile_r, inp_tile_c = _extra_fetch_dim2_helper(pe_info['num_pe'], inp_size(1), inp_size(2))
                    inp_tile_t = 1
                else:
                    assert "wrong dimensions other than [t,m,k]"

            elif pe_dataflow['x']['inter_pe'][0] == 'k':
                if pe_dataflow['x']['inter_pe'][1] == 't':
                    inp_tile_c, inp_tile_t = _extra_fetch_dim2_helper(pe_info['num_pe'], inp_size(2), inp_size(0))
                    inp_tile_r = 1
                elif pe_dataflow['x']['inter_pe'][1] == 'm':
                    inp_tile_c, inp_tile_r = _extra_fetch_dim2_helper(pe_info['num_pe'], inp_size(2), inp_size(1))
                    inp_tile_t = 1
                else:
                    assert "wrong dimensions other than [t,m,k]"
            else:
                assert "wrong dimensions other than [t,m,k]"

        #? Following cases allow 3 dimensions in inter_pe.
        elif len(pe_dataflow['x']['inter_pe']) == 3:
            if pe_dataflow['x']['inter_pe'][0]=='t':
                if pe_dataflow['x']['inter_pe'][1]=='m':
                    inp_tile_t, inp_tile_r, inp_tile_c = _extra_fetch_dim3_helper(pe_info['num_pe'], inp_size(0), inp_size(1), inp_size(2))
                elif pe_dataflow['x']['inter_pe'][1]=='k':
                    inp_tile_t, inp_tile_c, inp_tile_r = _extra_fetch_dim3_helper(pe_info['num_pe'], inp_size(0), inp_size(2), inp_size(1))
                else:
                    assert "wrong dimensions other than [t,m,k]"
            elif pe_dataflow['x']['inter_pe'][0]=='m':
                if pe_dataflow['x']['inter_pe'][1]=='t':
                    inp_tile_r, inp_tile_t, inp_tile_c = _extra_fetch_dim3_helper(pe_info['num_pe'], inp_size(1), inp_size(0), inp_size(2))
                elif pe_dataflow['x']['inter_pe'][1]=='k':
                    inp_tile_r, inp_tile_c, inp_tile_t = _extra_fetch_dim3_helper(pe_info['num_pe'], inp_size(1), inp_size(2), inp_size(0))
                else:
                    assert "wrong dimensions other than [t,m,k]"
            elif pe_dataflow['x']['inter_pe'][0]=='k':
                if pe_dataflow['x']['inter_pe'][1]=='t':
                    inp_tile_c, inp_tile_t, inp_tile_r = _extra_fetch_dim3_helper(pe_info['num_pe'], inp_size(2), inp_size(0), inp_size(1))
                elif pe_dataflow['x']['inter_pe'][1]=='m':
                    inp_tile_c, inp_tile_r, inp_tile_t = _extra_fetch_dim3_helper(pe_info['num_pe'], inp_size(2), inp_size(1), inp_size(0))
                else:
                    assert "wrong dimensions other than [t,m,k]"
        else:
            assert "wrong # of input dimensions, more than 3 or less than 1."

    #? A common case, allowing parallel PEs to receive different datas at same time.
    elif pe_dataflow['x']['intra_pe'] != 0 and pe_dataflow['x']['inter_pe'] != 0:
        #! TODO: currently, does not allow the overallping of dimensions between inter and intra.

        if len(pe_dataflow['x']['inter_pe']) == 1 and len(pe_dataflow['x']['intra_pe']) == 1:
            if pe_dataflow['x']['inter_pe'] == 't':
                inp_tile_t = _check_boundary_helper(pe_info['num_pe'], inp_size(0))
                if pe_dataflow['x']['intra_pe'] == 'm':
                    inp_tile_r = _check_boundary_helper(pe_info['num_inp'], inp_size(1))
                    inp_tile_c = 1
                elif pe_dataflow['x']['intra_pe'] == 'k':
                    inp_tile_c = _check_boundary_helper(pe_info['num_inp'], inp_size(2))
                    inp_tile_r = 1
                else:
                    assert "Wrong input intra dimensions other than t,m,k is provided."
            
            elif pe_dataflow['x']['inter_pe'] == 'm':
                inp_tile_r = _check_boundary_helper(pe_info['num_pe'], inp_size(1))
                if pe_dataflow['x']['intra_pe'] == 't':
                    inp_tile_t = _check_boundary_helper(pe_info['num_inp'], inp_size(0))
                    inp_tile_c = 1
                elif pe_dataflow['x']['intra_pe'] == 'k':
                    inp_tile_c = _check_boundary_helper(pe_info['num_inp'], inp_size(2))
                    inp_tile_t = 1
                else:
                    assert "Wrong input intra dimensions other than t,m,k is provided."

            elif pe_dataflow['x']['inter_pe'] == 'k':
                inp_tile_c = _check_boundary_helper(pe_info['num_pe'], inp_size(2))
                if pe_dataflow['x']['intra_pe'] == 't':
                    inp_tile_t = _check_boundary_helper(pe_info['num_inp'], inp_size(0))
                    inp_tile_r = 1
                elif pe_dataflow['x']['intra_pe'] == 'm':
                    inp_tile_r = _check_boundary_helper(pe_info['num_inp'], inp_size(1))
                    inp_tile_t = 1
                else:
                    assert "Wrong input intra dimensions other than t,m,k is provided."
            else:
                assert "Wrong input inter dimensions other than t,m,k is provided."
        
        #? We allow one of the inter or intra to have more than one dimensions as long as it does not overlap.
        elif len(pe_dataflow['x']['inter_pe']) == 1 and len(pe_dataflow['x']['intra_pe']) == 2:
            if pe_dataflow['x']['inter_pe'] == 't':
                inp_tile_t = _check_boundary_helper(pe_info['num_pe'], inp_size(0))
                if pe_dataflow['x']['intra_pe'] == 'mk':
                    inp_tile_r, inp_tile_c = _extra_fetch_dim2_helper(pe_info['num_inp'], inp_size(1), inp_size(2))
                elif pe_dataflow['x']['intra_pe'] == 'km':
                    inp_tile_c, inp_tile_r = _extra_fetch_dim2_helper(pe_info['num_inp'], inp_size(2), inp_size(1))
                else:
                    assert "Wrong input intra dimensions other than mk and km."
                    
            elif pe_dataflow['x']['inter_pe'] == 'm':
                inp_tile_r = _check_boundary_helper(pe_info['num_pe'], inp_size(1))
                if pe_dataflow['x']['intra_pe'] == 'tk':
                    inp_tile_t, inp_tile_c = _extra_fetch_dim2_helper(pe_info['num_inp'], inp_size(0), inp_size(2))
                elif pe_dataflow['x']['intra_pe'] == 'kt':
                    inp_tile_c, inp_tile_t = _extra_fetch_dim2_helper(pe_info['num_inp'], inp_size(2), inp_size(0))
                else:
                    assert "Wrong input intra dimensions other than tk and kt."

            elif pe_dataflow['x']['inter_pe'] == 'k':
                inp_tile_c = _check_boundary_helper(pe_info['num_pe'], inp_size(2))
                if pe_dataflow['x']['intra_pe'] == 'tm':
                    inp_tile_t, inp_tile_r = _extra_fetch_dim2_helper(pe_info['num_inp'], inp_size(0), inp_size(1))
                elif pe_dataflow['x']['intra_pe'] == 'mt':
                    inp_tile_r, inp_tile_t = _extra_fetch_dim2_helper(pe_info['num_inp'], inp_size(1), inp_size(0))
                else:
                    assert "Wrong input intra dimensions other than tm and mt."
            
            else:
                assert "Wrong input inter dimensions other than t,m,k."

        elif len(pe_dataflow['x']['inter_pe']) == 2 and len(pe_dataflow['x']['intra_pe']) == 1:
            if pe_dataflow['x']['intra_pe'] == 't':
                inp_tile_t = _check_boundary_helper(pe_info['num_inp'], inp_size(0))
                if pe_dataflow['x']['inter_pe'] == 'mk':
                    inp_tile_r, inp_tile_c = _extra_fetch_dim2_helper(pe_info['num_pe'], inp_size(1), inp_size(2))
                elif pe_dataflow['x']['inter_pe'] == 'km':
                    inp_tile_c, inp_tile_r = _extra_fetch_dim2_helper(pe_info['num_pe'], inp_size(2), inp_size(1))
                else:
                    assert "Wrong input inter dimensions other than mk and km."
                    
            elif pe_dataflow['x']['intra_pe'] == 'm':
                inp_tile_r = _check_boundary_helper(pe_info['num_inp'], inp_size(1))
                if pe_dataflow['x']['inter_pe'] == 'tk':
                    inp_tile_t, inp_tile_c = _extra_fetch_dim2_helper(pe_info['num_pe'], inp_size(0), inp_size(2))
                elif pe_dataflow['x']['inter_pe'] == 'kt':
                    inp_tile_c, inp_tile_t = _extra_fetch_dim2_helper(pe_info['num_pe'], inp_size(2), inp_size(0))
                else:
                    assert "Wrong input inter dimensions other than tk and kt."

            elif pe_dataflow['x']['intra_pe'] == 'k':
                inp_tile_c = _check_boundary_helper(pe_info['num_inp'], inp_size(2))
                if pe_dataflow['x']['inter_pe'] == 'tm':
                    inp_tile_t, inp_tile_r = _extra_fetch_dim2_helper(pe_info['num_pe'], inp_size(0), inp_size(1))
                elif pe_dataflow['x']['inter_pe'] == 'mt':
                    inp_tile_r, inp_tile_t = _extra_fetch_dim2_helper(pe_info['num_pe'], inp_size(1), inp_size(0))
                else:
                    assert "Wrong input inter dimensions other than tm and mt."
            
            else:
                assert "Wrong input intra dimensions other than t,m,k."
        
        else:
            assert "TODO: to be implemented for overlapping the dimensions between inter and intra."
    
    else:
        assert "TODO: support both inter and intra to be disabled. However, this should be very trivial and uncommon case."

    return (inp_tile_t, inp_tile_r, inp_tile_c)




def _Tile_to_pe_distr(pe_info, pe_dataflow, inp_tile, w_tile):

    n_pe = pe_info['num_pe']
    pe_w_lists = [None]*n_pe
    pe_x_lists = [None]*n_pe

    #? Step 1: Get inter-pe dimensions.
    w_inter_pe_dim = pe_dataflow['w']['inter_pe']
    w_intra_pe_dim = pe_dataflow['w']['intra_pe']

    #? 1. distribute the weight tiles:
    if w_inter_pe_dim == 0:
        pe_w_lists = [w_tile] * n_pe
    elif len(w_inter_pe_dim) == 1:
        if w_inter_pe_dim == 'k':
            for i in range(w_tile.size(0)):
                pe_w_lists[i] = w_tile[i,:]
        elif w_inter_pe_dim == 'n':
            for i in range(w_tile.size(1)):
                pe_w_lists[i] = w_tile[:,i]
        else:
            assert "wrong inter-pe dim for w"
    elif len(w_inter_pe_dim) == 2:
        if w_intra_pe_dim != 0:
            assert "TODO:, support multi dimensions at inter and intra level."
        else:
            if w_inter_pe_dim == 'kn':
                for i in range(w_tile.size(1)):
                    for j in range(w_tile.size(0)):
                        pe_w_lists[i*w_tile.size(0)+j] = w_tile[j,i]
            elif w_inter_pe_dim == 'nk':
                for i in range(w_tile.size(0)):
                    for j in range(w_tile.size(1)):
                        pe_w_lists[i*w_tile.size(1)+j] = w_tile[i,j]
            else:
                assert "wrong inter-pe dim for w"
    else:
        assert "wrong inter-pe dim number for w."


    #? Step 2: Get inter-pe dimensions for inp.
    x_inter_pe_dim = pe_dataflow['x']['inter_pe']
    x_intra_pe_dim = pe_dataflow['x']['intra_pe']
    x_dim_dict = {'t':0,'m':1,'k':2}

    #? 2. distribute the input tiles:
    if x_inter_pe_dim == 0:
        pe_x_lists = [inp_tile] * n_pe
    elif len(x_inter_pe_dim) == 1:
        if x_inter_pe_dim == 't':
            for i in range(inp_tile.size(0)):
                pe_x_lists[i] = inp_tile[i,:,:]
        elif x_inter_pe_dim == 'm':
            for i in range(inp_tile.size(1)):
                pe_x_lists[i] = inp_tile[:,i,:]
        elif x_inter_pe_dim == 'k':
            for i in range(inp_tile.size(2)):
                pe_x_lists[i] = inp_tile[:,:,i]
        else:
            assert "wrong inter-pe dim for inputs."
    elif len(x_inter_pe_dim) == 2:
    
        if x_inter_pe_dim == 'kt':
            for i in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])):
                for j in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])):
                    pe_x_lists[i*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+j] = inp_tile[i,:,j]
        if x_inter_pe_dim == 'tk':
            for i in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])):
                for j in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])):
                    pe_x_lists[i*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+j] = inp_tile[j,:,i]

        if x_inter_pe_dim == 'mt':
            for i in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])):
                for j in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])):
                    pe_x_lists[i*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+j] = inp_tile[i,j,:]
        if x_inter_pe_dim == 'tm':
            for i in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])):
                for j in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])):
                    pe_x_lists[i*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+j] = inp_tile[j,i,:]

        if x_inter_pe_dim == 'mk':
            for i in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])):
                for j in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])):
                    pe_x_lists[i*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+j] = inp_tile[:,j,i]
        if x_inter_pe_dim == 'km':
            for i in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])):
                for j in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])):
                    pe_x_lists[i*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+j] = inp_tile[:,i,j]


    elif len(x_inter_pe_dim) == 3:
        if x_inter_pe_dim == 'tkm':
            for i in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[2]])):
                    for j in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])):
                        for l in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])):
                            pe_x_lists[i*inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+j*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+l] = inp_tile[l,i,j]

        if x_inter_pe_dim == 'tmk':
            for i in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[2]])):
                    for j in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])):
                        for l in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])):
                            pe_x_lists[i*inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+j*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+l] = inp_tile[l,j,i]


        if x_inter_pe_dim == 'mtk':
            for i in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[2]])):
                    for j in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])):
                        for l in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])):
                            pe_x_lists[i*inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+j*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+l] = inp_tile[j,l,i]

        if x_inter_pe_dim == 'mkt':
            for i in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[2]])):
                    for j in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])):
                        for l in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])):
                            pe_x_lists[i*inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+j*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+l] = inp_tile[i,l,j]

        if x_inter_pe_dim == 'ktm':
            for i in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[2]])):
                    for j in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])):
                        for l in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])):
                            pe_x_lists[i*inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+j*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+l] = inp_tile[j,i,l]

        if x_inter_pe_dim == 'kmt':
            for i in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[2]])):
                    for j in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])):
                        for l in range(inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])):
                            pe_x_lists[i*inp_tile.size(x_dim_dict[x_inter_pe_dim[1]])*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+j*inp_tile.size(x_dim_dict[x_inter_pe_dim[0]])+l] = inp_tile[i,j,l]

    else:
        assert "Wrong input dimension numbers of inter-pe."

    return pe_x_lists, pe_w_lists






def _test_():

    def _test_Tile_shape_Anl_W():
        
        w = torch.randint(0,3,(4,4))
        pe_info = {
            'num_pe': 2,
            'num_inp': 1,
            'num_w': 2}
        print("--- original W ---")
        print(w)

        print('\n')
        print("---- BASIC TEST ----")
        pe_dataflow = {'w': {
            'inter_pe': 0,
            'intra_pe': 'k'}}
        w_tile_r, w_tile_c = _Tile_shape_Anl_W(pe_info, pe_dataflow, w.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(w[0:w_tile_r, 0:w_tile_c])

        pe_dataflow = {'w': {
            'inter_pe': 'k',
            'intra_pe': 0}}
        w_tile_r, w_tile_c = _Tile_shape_Anl_W(pe_info, pe_dataflow, w.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(w[0:w_tile_r, 0:w_tile_c])

        pe_dataflow = {'w': {
            'inter_pe': 0,
            'intra_pe': 'n'}}
        w_tile_r, w_tile_c = _Tile_shape_Anl_W(pe_info, pe_dataflow, w.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(w[0:w_tile_r, 0:w_tile_c])

        pe_dataflow = {'w': {
            'inter_pe': 'n',
            'intra_pe': 0}}
        w_tile_r, w_tile_c = _Tile_shape_Anl_W(pe_info, pe_dataflow, w.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(w[0:w_tile_r, 0:w_tile_c])

        print('\n')
        print("---- BASIC TEST (No Extra Fetch)----")
        pe_info = {
            'num_pe': 2,
            'num_inp': 1,
            'num_w': 6}
        pe_dataflow = {'w': {
            'inter_pe': 0,
            'intra_pe': 'n'}}
        w_tile_r, w_tile_c = _Tile_shape_Anl_W(pe_info, pe_dataflow, w.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(w[0:w_tile_r, 0:w_tile_c])

        pe_info = {
            'num_pe': 2,
            'num_inp': 1,
            'num_w': 6}
        pe_dataflow = {'w': {
            'inter_pe': 0,
            'intra_pe': 'k'}}
        w_tile_r, w_tile_c = _Tile_shape_Anl_W(pe_info, pe_dataflow, w.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(w[0:w_tile_r, 0:w_tile_c])

        pe_info = {
            'num_pe': 6,
            'num_inp': 1,
            'num_w': 6}
        pe_dataflow = {'w': {
            'inter_pe': 'k',
            'intra_pe': 0}}
        w_tile_r, w_tile_c = _Tile_shape_Anl_W(pe_info, pe_dataflow, w.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(w[0:w_tile_r, 0:w_tile_c])

        print('\n')
        print("---- BASIC TEST (Extra Fetch)----")

        pe_info = {
            'num_pe': 9,
            'num_inp': 1,
            'num_w': 6}
        pe_dataflow = {'w': {
            'inter_pe': 'kn',
            'intra_pe': 0}}
        w_tile_r, w_tile_c = _Tile_shape_Anl_W(pe_info, pe_dataflow, w.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(w[0:w_tile_r, 0:w_tile_c])

        pe_info = {
            'num_pe': 4,
            'num_inp': 1,
            'num_w': 8}
        pe_dataflow = {'w': {
            'inter_pe': 0,
            'intra_pe': 'nk'}}
        w_tile_r, w_tile_c = _Tile_shape_Anl_W(pe_info, pe_dataflow, w.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(w[0:w_tile_r, 0:w_tile_c])


        print('\n')
        print("---- ADVANCED TEST (Parallel and Multi-local)----")

        pe_info = {
            'num_pe': 2,
            'num_inp': 1,
            'num_w': 4}
        pe_dataflow = {'w': {
            'inter_pe': 'k',
            'intra_pe': 'n'}}
        w_tile_r, w_tile_c = _Tile_shape_Anl_W(pe_info, pe_dataflow, w.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(w[0:w_tile_r, 0:w_tile_c])

        pe_info = {
            'num_pe': 2,
            'num_inp': 1,
            'num_w': 4}
        pe_dataflow = {'w': {
            'inter_pe': 'n',
            'intra_pe': 'k'}}
        w_tile_r, w_tile_c = _Tile_shape_Anl_W(pe_info, pe_dataflow, w.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(w[0:w_tile_r, 0:w_tile_c])

        pe_info = {
            'num_pe': 3,
            'num_inp': 1,
            'num_w': 4}
        pe_dataflow = {'w': {
            'inter_pe': 'n',
            'intra_pe': 'k'}}
        w_tile_r, w_tile_c = _Tile_shape_Anl_W(pe_info, pe_dataflow, w.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(w[0:w_tile_r, 0:w_tile_c])


    def _test_Tile_shape_Anl_X():
        
        x = torch.randint(0,4,(2,4,4))
        pe_info = {
            'num_pe': 2,
            'num_inp': 4,
            'num_w': 2}
        print("--- original X ---")
        print(x)

        print('\n')
        print("---- BASIC TEST ----")
        pe_dataflow = {'x': {
            'inter_pe': 0,
            'intra_pe': 't'}}
        x_tile_t, x_tile_r, x_tile_c = _Tile_shape_Anl_Inp(pe_info, pe_dataflow, x.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(x[0:x_tile_t, 0:x_tile_r, 0:x_tile_c])

        pe_dataflow = {'x': {
            'inter_pe': 't',
            'intra_pe': 0}}
        x_tile_t, x_tile_r, x_tile_c = _Tile_shape_Anl_Inp(pe_info, pe_dataflow, x.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(x[0:x_tile_t, 0:x_tile_r, 0:x_tile_c])

        pe_dataflow = {'x': {
            'inter_pe': 0,
            'intra_pe': 'm'}}
        x_tile_t, x_tile_r, x_tile_c = _Tile_shape_Anl_Inp(pe_info, pe_dataflow, x.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(x[0:x_tile_t, 0:x_tile_r, 0:x_tile_c])

        pe_dataflow = {'x': {
            'inter_pe': 'm',
            'intra_pe': 0}}
        x_tile_t, x_tile_r, x_tile_c = _Tile_shape_Anl_Inp(pe_info, pe_dataflow, x.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(x[0:x_tile_t, 0:x_tile_r, 0:x_tile_c])

        pe_dataflow = {'x': {
            'inter_pe': 0,
            'intra_pe': 'k'}}
        x_tile_t, x_tile_r, x_tile_c = _Tile_shape_Anl_Inp(pe_info, pe_dataflow, x.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(x[0:x_tile_t, 0:x_tile_r, 0:x_tile_c])

        pe_dataflow = {'x': {
            'inter_pe': 'k',
            'intra_pe': 0}}
        x_tile_t, x_tile_r, x_tile_c = _Tile_shape_Anl_Inp(pe_info, pe_dataflow, x.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(x[0:x_tile_t, 0:x_tile_r, 0:x_tile_c])

        print('\n')
        print("---- BASIC TEST 2 DIM----")
        
        pe_info = {
            'num_pe': 8,
            'num_inp': 8,
            'num_w': 0}
        pe_dataflow = {'x': {
            'inter_pe': 0,
            'intra_pe': 'km'}}
        x_tile_t, x_tile_r, x_tile_c = _Tile_shape_Anl_Inp(pe_info, pe_dataflow, x.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(x[0:x_tile_t, 0:x_tile_r, 0:x_tile_c])


        print('\n')
        print("---- ADVANCED TEST 3 DIM----")
        
        pe_info = {
            'num_pe': 8,
            'num_inp': 8,
            'num_w': 0}
        pe_dataflow = {'x': {
            'inter_pe': 'k',
            'intra_pe': 'tm'}}
        x_tile_t, x_tile_r, x_tile_c = _Tile_shape_Anl_Inp(pe_info, pe_dataflow, x.size)
        print("pe_dataflow: ")
        print(pe_dataflow)
        print("--- first tile ---")
        print(x[0:x_tile_t, 0:x_tile_r, 0:x_tile_c])



    def _test_Tile_PE_distribute():
        x = torch.randint(0,4,(4,6,6))
        print("--- original X ---")
        print(x)
        print("\n")
        w = torch.randint(0,4,(6,6))
        print("--- original W ---")
        print(w)
        print("\n")

        pe_info = {
            'num_pe': 4,
            'num_inp': 24,
            'num_w': 6}
        print(f"Number of PEs: {pe_info['num_pe']}")
        print(f"Each PE holds {pe_info['num_w']} weights")
        print(f"Each PE holds {pe_info['num_inp']} inputs")
        print("\n")

        pe_dataflow = {'x': {
            'inter_pe': 0,
            'intra_pe': 'tk'},
            'w': {
            'inter_pe': 'n',
            'intra_pe': 'k'}}
        print(f"PE Dataflow: {pe_dataflow}")
        print("\n")
        
        
        w_tile_r, w_tile_c = _Tile_shape_Anl_W(pe_info, pe_dataflow, w.size)
        x_tile_t, x_tile_r, x_tile_c = _Tile_shape_Anl_Inp(pe_info, pe_dataflow, x.size)

        w_tile = w[0:w_tile_r, 0:w_tile_c]
        x_tile = x[0:x_tile_t, 0:x_tile_r, 0:x_tile_c]

        x_pe, w_pe = _Tile_to_pe_distr(pe_info, pe_dataflow, x_tile, w_tile)

        for p in range(len(x_pe)):
            print(f'{p}th PE holds the input tile below: ')
            print(x_pe[p])
            print("\n")

        for p in range(len(w_pe)):
            print(f'{p}th PE holds the weight tile below: ')
            print(w_pe[p])
            print("\n")
        

    # _test_Tile_shape_Anl_W()
    # _test_Tile_shape_Anl_X()
    _test_Tile_PE_distribute()


def main():
    _test_()

if __name__ == "__main__":
    main()