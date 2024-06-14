# ===============================================================
# useful helper functions that are commonly used in estimators
# Source: Accelergy -- https://github.com/Accelergy-Project/accelergy/blob/master/accelergy/helper_functions.py
# ===============================================================



# ===============================================================
# Format of the known_result 
#
# - bw: x0
#   attributes:
#       area: 0.0024563018
#       power-dynamic: 0.239484
#       power-leakage: 0.057968
#       access-latency: 1
#       cycle-time: 2.5
#
# - bw: x1
#   attributes:
#       area: ...
# ===============================================================

def oneD_linear_interpolation(desired_bw, attr_key, known_result):
    # assume E = ax + c where x is a hardware attribute
    ordered_list = []
    if known_result[1]['bw'] < known_result[0]['bw']: #* This is to make sure the interpolation is done with an ascending bw.
        ordered_list.append(known_result[1])
        ordered_list.append(known_result[0])
    else:
        ordered_list = known_result

    slope = (known_result[1]['attributes'][attr_key] - known_result[0]['attributes'][attr_key]) / (known_result[1]['bw'] - known_result[0]['bw'])
    desired_result = slope * (desired_bw - ordered_list[0]['bw']) + ordered_list[0]['attributes'][attr_key]
    return desired_result

def oneD_quadratic_interpolation(desired_bw, attr_key, known_result):
    # assume E = ax^2 + c where x is a hardware attribute
    ordered_list = []
    if known_result[1]['bw'] < known_result[0]['bw']: #* This is to make sure the interpolation is done with an ascending bw.
        ordered_list.append(known_result[1])
        ordered_list.append(known_result[0])
    else:
        ordered_list = known_result

    slope = (known_result[1]['attributes'][attr_key] - known_result[0]['attributes'][attr_key]) / (known_result[1]['bw']**2 - known_result[0]['bw']**2)
    desired_result = slope * (desired_bw**2 - ordered_list[0]['bw']**2) + ordered_list[0]['attributes'][attr_key]
    return desired_result

# ===============================================================
# useful helper functions for sparse worklaods and conv2d->matmul
# ===============================================================
def _im2col(inp_tensor, w_tensor, padding=0, stride=1, dilation=1):
    return F.unfold(inp_tensor,w_tensor.shape[2:4], dilation, padding, stride)

def _conv2matmul(input_tensor, w_tensor, padding=0, stride=1, dilation=1):
    inp_converted = _im2col(input_tensor, w_tensor, padding, stride, dilation).transpose(1,2)
    w_converted = w_tensor.view(w_tensor.size(0),-1).t()
    return inp_converted, w_converted

def _value2mask(matrix):
    return (matrix != 0).float()