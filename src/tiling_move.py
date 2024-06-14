import torch
import time
import math
import torch.nn.functional as F
from tqdm import tqdm

from tiling_gen import _Tile_shape_Anl_W, _Tile_shape_Anl_Inp, _Tile_to_pe_distr

def _Tile_movement_Custom(inp_mat, w_mat, dataflow, tile_dict=None):
    
    inp_pe_t, inp_pe_r, inp_pe_c = tile_dict['x']
    w_pe_r, w_pe_c = tile_dict['w']

    #* Bound checkin for the custom movement.
    def bound_check(move_dict):
        if move_dict['inp_t_t'] >= inp_mat.size(0):
            move_dict['inp_t_t'] = inp_mat.size(0)
        if move_dict['inp_r_t'] >= inp_mat.size(1):
            move_dict['inp_r_t'] = inp_mat.size(1)
        if move_dict['inp_c_t'] >= inp_mat.size(2):
            move_dict['inp_c_t'] = inp_mat.size(2)
        if move_dict['w_r_t'] >= w_mat.size(0):
            move_dict['w_r_t'] = w_mat.size(0)
        if move_dict['w_c_t'] >= w_mat.size(1):
            move_dict['w_c_t'] = w_mat.size(1)
        return move_dict
    
    #* Depending on the loop index, move the tiling window.
    def decode_tile_move(move_dict, dim, indx):
        if dim == 't':
            move_dict['inp_t_h'] = indx*inp_pe_t
            move_dict['inp_t_t'] = (indx+1)*inp_pe_t
        elif dim == 'm':
            move_dict['inp_r_h'] = indx*inp_pe_r
            move_dict['inp_r_t'] = (indx+1)*inp_pe_r
        elif dim == 'n':
            move_dict['w_c_h'] = indx*w_pe_c
            move_dict['w_c_t'] = (indx+1)*w_pe_c
        elif dim == 'k':
            move_dict['w_r_h'] = indx*w_pe_r
            move_dict['w_r_t'] = (indx+1)*w_pe_r
            move_dict['inp_c_h'] = indx*inp_pe_c
            move_dict['inp_c_t'] = (indx+1)*inp_pe_c
        else:
            assert "dataflow has dimensions other than [t,k,m,n]"
        return move_dict

    #! Initialize the tile lists.
    inp_tile_lists = []
    w_tile_lists = []
    #? Initialize the position index with one more time dimension (inp_t_h).
    move_dict = {'inp_t_h': 0,
                 'inp_r_h': 0,
                 'inp_c_h': 0,
                 'w_r_h': 0,
                 'w_c_h': 0,
                 'inp_t_t': inp_pe_t,
                 'inp_r_t': inp_pe_r,
                 'inp_c_t': inp_pe_c,
                 'w_r_t': w_pe_r,
                 'w_c_t': w_pe_c}

    #! Decide the moving sizes for SNN.
    t_steps = math.ceil(inp_mat.size(0)/inp_pe_t) #? Steps need to move along t-dim.
    m_steps = math.ceil(inp_mat.size(1)/inp_pe_r) #? Steps need to move along m-dim.
    n_steps = math.ceil(w_mat.size(1)/w_pe_c) #? Steps need to move along n-dim.
    k_steps = math.ceil(w_mat.size(0)/w_pe_r) #? Steps need to move along k-dim.

    assert k_steps == math.ceil(inp_mat.size(2)/inp_pe_c), "x and w tile does not share the k dimension move, not supported!"
    step_dict = {'m': m_steps,
                    'n': n_steps,
                    'k': k_steps,
                    't': t_steps}

    #? Main 4-nested loops for the tile to move along the matrices.
    for i in range(step_dict[dataflow[0]]):
        move_dict = decode_tile_move(move_dict, dataflow[0], i)
        for j in range(step_dict[dataflow[1]]):
            move_dict = decode_tile_move(move_dict, dataflow[1], j)
            for k in range(step_dict[dataflow[2]]):
                move_dict = decode_tile_move(move_dict, dataflow[2], k)
                for l in range(step_dict[dataflow[3]]):
                    move_dict = decode_tile_move(move_dict, dataflow[3], l)
                    #! Make sure the index is in-bound for the matrix.
                    move_dict = bound_check(move_dict)

                    inp_tile_lists += [inp_mat[move_dict['inp_t_h']:move_dict['inp_t_t'], move_dict['inp_r_h']:move_dict['inp_r_t'], move_dict['inp_c_h']:move_dict['inp_c_t']]]
                    w_tile_lists += [w_mat[move_dict['w_r_h']:move_dict['w_r_t'], move_dict['w_c_h']:move_dict['w_c_t']]]

    return inp_tile_lists, w_tile_lists

def _Tile_movement_Anl(inp_mat, w_mat, pe_info, pe_dataflow, dataflow):

    #! This is the tile shape.
    inp_pe_t, inp_pe_r, inp_pe_c = _Tile_shape_Anl_Inp(pe_info, pe_dataflow, inp_mat.size)
    w_pe_r, w_pe_c = _Tile_shape_Anl_W(pe_info, pe_dataflow, w_mat.size)

    #* Bound checkin for non-SNN tiling.
    def bound_check(move_dict):  
        if move_dict['inp_t_t'] >= inp_mat.size(0):
            move_dict['inp_t_t'] = inp_mat.size(0)
        if move_dict['inp_c_t'] >= inp_mat.size(1):
            move_dict['inp_c_t'] = inp_mat.size(1)
        if move_dict['w_r_t'] >= w_mat.size(0):
            move_dict['w_r_t'] = w_mat.size(0)
        if move_dict['w_c_t'] >= w_mat.size(1):
            move_dict['w_c_t'] = w_mat.size(1)
        return move_dict

    #* Depending on the loop index, move the tiling window.
    def decode_tile_move(move_dict, dim, indx):
        if dim == 't':
            move_dict['inp_t_h'] = indx*inp_pe_t
            move_dict['inp_t_t'] = (indx+1)*inp_pe_t
        elif dim == 'm':
            move_dict['inp_r_h'] = indx*inp_pe_r
            move_dict['inp_r_t'] = (indx+1)*inp_pe_r
        elif dim == 'n':
            move_dict['w_c_h'] = indx*w_pe_c
            move_dict['w_c_t'] = (indx+1)*w_pe_c
        elif dim == 'k':
            move_dict['w_r_h'] = indx*w_pe_r
            move_dict['w_r_t'] = (indx+1)*w_pe_r
            move_dict['inp_c_h'] = indx*inp_pe_c
            move_dict['inp_c_t'] = (indx+1)*inp_pe_c
        else:
            assert "dataflow has dimensions other than [t,k,m,n]"
        return move_dict
    
    #! Initialize the tile lists.
    inp_tile_lists = []
    w_tile_lists = []

    #! This indicates the SNN mode.
    if 't' in dataflow: 
        #? Initialize the position index with one more time dimension (inp_t_h).
        move_dict = {'inp_t_h': 0,
                     'inp_r_h': 0,
                     'inp_c_h': 0,
                     'w_r_h': 0,
                     'w_c_h': 0,
                     'inp_t_t': inp_pe_t,
                     'inp_r_t': inp_pe_r,
                     'inp_c_t': inp_pe_c,
                     'w_r_t': w_pe_r,
                     'w_c_t': w_pe_c}

        #! Decide the moving sizes for SNN.
        timesteps = inp_mat.size(0)
        t_steps = math.ceil(timesteps/inp_pe_t) #? Steps need to move along t-dim.
        m_steps = math.ceil(inp_mat.size(1)/inp_pe_r) #? Steps need to move along m-dim.
        n_steps = math.ceil(w_mat.size(1)/w_pe_c) #? Steps need to move along n-dim.
        k_steps = math.ceil(w_mat.size(0)/w_pe_r) #? Steps need to move along k-dim.

        if k_steps != inp_mat.size(1)/inp_pe_c:
            assert "x and w tile does not share the k dimension move, not supported!"
        

        #! Check whether steps have tails here.
        #! TODO: more complicated case is that on k dimension, x and w move different steps.
        #! TODO: neglect this case for now.

        step_dict = {'m': m_steps,
                     'n': n_steps,
                     'k': k_steps,
                     't': t_steps}

        #? Since the temporal dimensions, there are 4-loop nests.
        for i in range(step_dict[dataflow[0]]):
            move_dict = decode_tile_move(move_dict, dataflow[0], i)
            for j in range(step_dict[dataflow[1]]):
                move_dict = decode_tile_move(move_dict, dataflow[1], j)
                for k in range(step_dict[dataflow[2]]):
                    move_dict = decode_tile_move(move_dict, dataflow[2], k)
                    for l in range(step_dict[dataflow[3]]):
                        move_dict = decode_tile_move(move_dict, dataflow[3], l)
                        inp_tile_lists += [inp_mat[move_dict['inp_t_h']:move_dict['inp_t_t'], move_dict['inp_r_h']:move_dict['inp_r_t'], move_dict['inp_c_h']:move_dict['inp_c_t']]]
                        w_tile_lists += [w_mat[move_dict['w_r_h']:move_dict['w_r_t'], move_dict['w_c_h']:move_dict['w_c_t']]]
    # #! This is the ANN mode.
    # else:
    #     #? Initialize the position index for the non-SNN input and weight matrix.
    #     move_dict = {'inp_r_h': 0,
    #                  'inp_c_h': 0,
    #                  'w_r_h': 0,
    #                  'w_c_h': 0,
    #                  'inp_r_t': inp_pe_r,
    #                  'inp_c_t': inp_pe_c,
    #                  'w_r_t': w_pe_r,
    #                  'w_c_t': w_pe_c}
    #     move_dict = bound_check(move_dict) #! Important to check the bound for not overflowin.

    #     #! Decide the moving sizes for ANN.
    #     m_steps = math.ceil(inp_mat.size(0)/inp_pe_r) #? Steps need to move along m-dim.
    #     n_steps = math.ceil(w_mat.size(1)/w_pe_c) #? Steps need to move along n-dim.
    #     k_steps = math.ceil(w_mat.size(0)/w_pe_r) #? Steps need to move along k-dim.

    #     step_dict = {'m': m_steps,
    #                  'n': n_steps,
    #                  'k': k_steps}

    #     #? Standard 3-loop nests.
    #     for i in range(step_dict[dataflow[0]]):
    #         decode_tile_move(move_dict, dataflow[0], i)
    #         for j in range(step_dict[dataflow[1]]):
    #             decode_tile_move(move_dict, dataflow[1], j)
    #             for k in range(step_dict[dataflow[2]]):
    #                 decode_tile_move(move_dict, dataflow[2], k)
    #                 inp_tile_lists += [inp_mat[move_dict['inp_r_h']:move_dict['inp_r_t'], move_dict['inp_c_h']:move_dict['inp_c_t']]]
    #                 w_tile_lists += [w_mat[move_dict['w_r_h']:move_dict['w_r_t'], move_dict['w_c_h']:move_dict['w_c_t']]]
                        
    return inp_tile_lists, w_tile_lists



def test():

    def _Tile_movement_Custom_Test():
        print("\n")
        print("------ TEST CASE 1 ------")
        print("\n")

        print("Input matrix 4x8x8: ")
        inp = torch.randint(0,4,(4,8,8))
        print(inp)
        print("\n")
        print("Weight matrix 8x16: ")
        w = torch.randint(0,4,(8,16))
        print(w)
        print("\n")
        tiling_dataflow = 'nmtk'
        print(f"Tiling Dataflow: {tiling_dataflow}")

        tile_dict = {'x':(3,4,4),
                     'w':(4,4)}

        x_tiles, w_tiles = _Tile_movement_Custom(inp, w, tiling_dataflow, tile_dict)
        for i in range(len(x_tiles)):
                print(f'The {i}-th input tile: \n',x_tiles[i])
                print(f'The {i}-th weight tile: \n',w_tiles[i])

    def _Tile_movement_Anl_Test(test_num):
    
        if test_num == 1: 
            print("\n")
            print("------ TEST CASE 1 ------")
            print("\n")

            print("Input matrix 4x8x8: ")
            inp = torch.randint(0,4,(4,8,8))
            print(inp)
            print("\n")
            print("Weight matrix 8x16: ")
            w = torch.randint(0,4,(8,16))
            print(w)
            print("\n")

            pe_info = {"num_pe": 8,
                        "num_w": 8,
                        "num_inp": 8}
            
            print(f"Number of PEs: {pe_info['num_pe']}")
            print(f"Each PE holds {pe_info['num_w']} weights")
            print(f"Each PE holds {pe_info['num_inp']} inputs")

            pe_dataflow = {
                'x': {'inter_pe': 0,
                        'intra_pe': 'k'},
                'w': {'inter_pe': 'n',
                        'intra_pe': 'k'} 
            }
            print(f"PE Dataflow: {pe_dataflow}")

            tiling_dataflow = 'tnmk'
            print(f"Tiling Dataflow: {tiling_dataflow}")

            x_tiles, w_tiles = _Tile_movement_Anl(inp, w, pe_info, pe_dataflow, tiling_dataflow)

            for i in range(len(x_tiles)):
                print(f'The {i}-th input tile: \n',x_tiles[i])
                print(f'The {i}-th weight tile: \n',w_tiles[i])
        
        if test_num == 2:
            print("\n")
            print("------ TEST CASE 2 ------")
            print("\n")

            print("Input matrix 4x8x8: ")
            inp = torch.randint(0,4,(4,8,8))
            print(inp)
            print("\n")
            print("Weight matrix 8x16: ")
            w = torch.randint(0,4,(8,16))
            print(w)
            print("\n")

            pe_info = {"num_pe": 8,
                        "num_w": 8,
                        "num_inp": 8}
            
            print(f"Number of PEs: {pe_info['num_pe']}")
            print(f"Each PE holds {pe_info['num_w']} weights")
            print(f"Each PE holds {pe_info['num_inp']} inputs")

            pe_dataflow = {
                'x': {'inter_pe': 0,
                        'intra_pe': 'k'},
                'w': {'inter_pe': 'n',
                        'intra_pe': 'k'} 
            }
            print(f"PE Dataflow: {pe_dataflow}")

            tiling_dataflow = 'ntmk'
            print(f"Tiling Dataflow: {tiling_dataflow}")

            x_tiles, w_tiles = _Tile_movement_Anl(inp, w, pe_info, pe_dataflow, tiling_dataflow)

            for i in range(len(x_tiles)):
                print(f'The {i}-th input tile: \n',x_tiles[i])
                print(f'The {i}-th weight tile: \n',w_tiles[i])

    
    # _Tile_movement_Anl_Test(2)
    _Tile_movement_Custom_Test()

if __name__ == "__main__":
    test()