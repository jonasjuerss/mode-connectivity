import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn.modules.utils import _pair
from scipy.special import binom

#reimplementation of the curve module based on the provided code
#code segments copied without modifications for improvement are indicated by comments

#improved vram efficiency over original implementation
class Bezier():
    def __init__(self, num_bends):
        self.num_bends = num_bends

    def get_coeffs(self, t):
        return torch.tensor(binom(self.num_bends - 1, np.arange(self.num_bends), dtype=np.float32)) * \
               torch.pow(t, torch.arange(0, float(self.num_bends))) * \
               torch.pow((1.0 - t), torch.arange(float(self.num_bends - 1), -1, -1))

#improved vram efficiency over original implementation
class PolyChain(Module):
    def __init__(self, num_bends):
        self.num_bends = num_bends

    def get_coeffs(self, t):
        t_n = t * (self.num_bends - 1)
        return torch.max(torch.tensor(0.0), 1.0 - torch.abs(t_n - torch.arange(0, float(self.num_bends))))
        

class CurveSystem():
    """ This class is used to map a value in [0,1] to the corresponding point in the CurveSystem where the point 
    is given as the coefficients for the networks at each bend/endpoint (e.g. Point t=0.3 is 0.5*Network1 + 0.3 * Network2))

    Coefficients correspond as following (Middle point connecting all = M) [M, E1, E2, ..., En, E1-M:Bend1, E1-M:B2, ..., E2-M:B1, E2-M:B2, ..., En-M:B1 , ...]
    """
    def __init__(self, num_bends, num_end_points, curve):
        """
        Parameters:
            num_bends : number of bends in a single curve connecting an endpoint and the middle M (including start and endpoint)
            num_end_points : number of endpoints in the system
            curve : class of the curves used (e.g. PolyChain)
        """
        self.num_bends = num_bends
        self.num_end_points = num_end_points
        self.curve = curve(num_bends)
        self.num_coefficients =  1 + num_end_points + (num_bends-2) * num_end_points 

    def forward(self, t):
        t = t * self.num_end_points
        end_point = int(t // 1) #indicates the curve to which endpoint we are at (in [0, .., n_end_points - 1])
        t = t % 1 #indicates the t on this curve
        local_curve_coeff = self.curve(t)
        t_n = t * (self.num_bends - 1)
        result = torch.zeros(self.num_coefficients)
        result[0] = local_curve_coeff[0] #set M coeff
        result[end_point + 1] = local_curve_coeff[-1] #set endpoint coeff
        bend_indices_start = self.num_end_points + 1 + (end_point * (self.num_bends - 2))
        result[bend_indices_start : bend_indices_start + self.num_bends - 2] = local_curve_coeff[1:-1]

        return result




#improved compute_weights_t method over the original implementation
class CurveModule(Module):

    def __init__(self, fix_points, parameter_names=()):
        super(CurveModule, self).__init__()
        self.fix_points = fix_points
        self.num_bends = len(self.fix_points)
        self.parameter_names = parameter_names
        self.l2 = 0.0

    def compute_weights_t(self, coeffs_t):
        w_t = [None] * len(self.parameter_names)
        self.l2 = 0.0
        for i, parameter_name in enumerate(self.parameter_names):
            for j, coeff in enumerate(coeffs_t):
                if(coeff.item() == 0):
                    continue
                parameter = getattr(self, '%s_%d' % (parameter_name, j))
                if parameter is not None:
                    if w_t[i] is None:
                        w_t[i] = parameter * coeff
                    else:
                        w_t[i] += parameter * coeff
            if w_t[i] is not None:
                self.l2 += torch.sum(w_t[i] ** 2)
        return w_t

#improved vram efficiency over original implementation
class Linear(CurveModule):

    def __init__(self, in_features, out_features, fix_points, bias=True):
        super(Linear, self).__init__(fix_points, ('weight', 'bias'))
        self.in_features = in_features
        self.out_features = out_features

        self.l2 = 0.0
        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                'weight_%d' % i,
                Parameter((torch.rand((out_features, in_features),requires_grad=not fixed) - 0.5)* 2 / math.sqrt(self.in_features), requires_grad=not fixed)
            )
        for i, fixed in enumerate(self.fix_points):
            if bias:
                self.register_parameter(
                    'bias_%d' % i,
                    Parameter((torch.rand((out_features),requires_grad=not fixed) - 0.5)* 2 / math.sqrt(self.in_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('bias_%d' % i, None)

    def forward(self, input, coeffs_t):
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.linear(input, weight_t, bias_t)

#improved vram efficiency over original implementation
class Conv2d(CurveModule):

    def __init__(self, in_channels, out_channels, kernel_size, fix_points, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(fix_points, ('weight', 'bias'))
        if (in_channels % groups) != 0 or (out_channels % groups != 0):
            raise ValueError('both in_channels and out_channels must be divisible by groups')

        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        for i, fixed in enumerate(self.fix_points):
            self.register_parameter('weight_%d' % i, Parameter((torch.rand((out_channels, in_channels // groups, *kernel_size))- 0.5)* 2 * stdv, requires_grad=not fixed))
        for i, fixed in enumerate(self.fix_points):
            if bias:
                self.register_parameter('bias_%d' % i, Parameter((torch.rand((out_channels)) - 0.5)*2*stdv, requires_grad=not fixed))
            else:
                self.register_parameter('bias_%d' % i, None)

 
    def forward(self, input, coeffs_t):
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.conv2d(input, weight_t, bias_t, self.stride,
                        self.padding, self.dilation, self.groups)

class _BatchNorm(CurveModule):
    _version = 2

    def __init__(self, num_features, fix_points, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm, self).__init__(fix_points, ('weight', 'bias'))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.l2 = 0.0
        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                self.register_parameter(
                    'weight_%d' % i,
                    Parameter(torch.rand(num_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('weight_%d' % i, None)
        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                self.register_parameter(
                    'bias_%d' % i,
                    Parameter(torch.zeros(num_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('bias_%d' % i, None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_running_stats()

    #copied functions reset_running_stats, _check_input_dim forward, extra_repr, _load_from_state_dict from original implementation 
    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()


    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input, coeffs_t):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.batch_norm(
            input, self.running_mean, self.running_var, weight_t, bias_t,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)
    
    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_BatchNorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

#copied from original implementation
class BatchNorm2d(_BatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class CurveNet(Module):
    def __init__(self, num_classes, curve, architecture, num_bends, fix_start=True, fix_end=True,
                 architecture_kwargs={}):
        super(CurveNet, self).__init__()
        self.num_bends = num_bends
        self.num_classes = num_classes
        self.fix_points = [fix_start] + [False] * (self.num_bends - 2) + [fix_end]    
        self.architecture = architecture
        self.curve = curve(self.num_bends)
        self.l2 = 0.0
        self.curve_architecture = self.architecture(num_classes, fix_points=self.fix_points, **architecture_kwargs)
        self.curve_modules = []
        for module in self.curve_architecture.modules():
            if issubclass(module.__class__, CurveModule):
                self.curve_modules.append(module)

    def import_base_parameters(self, base_model, index):
        base_parameters = base_model.parameters()
        parameters = list(self.curve_architecture.parameters())[index::self.num_bends]
        for i in range(len(parameters)):
            parameters[i] = base_parameters[i].detach().clone()

    def import_base_buffers(self, base_model):
        curvenet_buffers = self.curve_architecture._all_buffers()
        base_model_buffers = base_model._all_buffers()
        for i in range(len(curvenet_buffers)):
            curvenet_buffers[i] = base_model_buffers[i].detach.clone()


    def export_base_parameters(self, base_model, index):
        base_parameters = base_model.parameters()
        parameters = list(self.curve_architecture.parameters())[index::self.num_bends]
        for i in range(len(parameters)):
            base_parameters[i] = parameters[i].detach().clone()

    def init_linear(self):
        parameters = list(self.curve_architecture.parameters())
        for i in range(0, len(parameters), self.num_bends):
            weights = parameters[i:i+self.num_bends]
            for j in range(1, self.num_bends - 1):
                alpha = j * 1.0 / (self.num_bends - 1)
                weights[j] = (alpha * weights[-1] + (1.0 - alpha) * weights[0]).detach().copy()

    #copied weights an _compute_l2 function from original implementation
    def weights(self, t):
        coeffs_t = self.curve.get_coeffs(t)
        weights = []
        for module in self.curve_modules:
            weights.extend([w for w in module.compute_weights_t(coeffs_t) if w is not None])
        return np.concatenate([w.detach().cpu().numpy().ravel() for w in weights])

    def _compute_l2(self):
        self.l2 = sum(module.l2 for module in self.curve_modules)

    def forward(self, input, t=None):
        if t is None:
            t = torch.rand(1)
        coeffs_t = self.curve.get_coeffs(t)
        output = self.curve_architecture(input, coeffs_t)
        self._compute_l2()
        return output

class CurveSystemNet(Module):
    def __init__(self, num_classes, curve, architecture, num_bends, fix_end_points = [True, True, True],
                 architecture_kwargs={}):
        """ Constructor for CurveSystemNet that connects a number of networks with a "tunnel system" consisting of multiple curves all running together in a single point
        
        Parameters:
            num_classes : number of classes to be predicted
            curve : Curve module for calculating coefficients (e.g. PolyChain)
            architecture : (Curve-) Network architecture of an endpoint (e.g. ConvFCCurve)
            num_bends : bends of a curve between an endpoint and the point where all the connections run together (including the start and end point -> 2 means straight line )
            fix_endpoints : which endpoints can be altered during training
            architecture_kwargs : architecture specific keywords (e.g. dropout rate)
 
        """

        if(n_end_points < 2):
            raise ValueError("CurveSystems require at least two endpoints")
            
        super(CurveSystemNet, self).__init__()
        n_end_points = len(fix_end_points)
        
        self.num_classes = num_classes
        self.num_bends = num_bends
        
        self.fix_points = [False] + fix_end_points + [False] * ((num_bends-2) * n_end_points) #curve connecting
        
        self.curve = curve(self.num_bends)
        self.architecture = architecture

        self.l2 = 0.0
        self.curve_architecture = self.architecture(num_classes, fix_points=self.fix_points, **architecture_kwargs) #eg MNISTNetCurve
        self.curve_modules = []
        for module in self.curve_architecture.modules():
            if issubclass(module.__class__, CurveModule):
                self.curve_modules.append(module)

    def import_base_parameters(self, base_model, index):
        parameters = list(self.curve_architecture.parameters())[index::self.num_bends]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            parameter.data.copy_(base_parameter.data)

    def import_base_buffers(self, base_model):
        for buffer, base_buffer in zip(self.curve_architecture._all_buffers(), base_model._all_buffers()):
            buffer.data.copy_(base_buffer.data)

    def export_base_parameters(self, base_model, index):
        parameters = list(self.curve_architecture.parameters())[index::self.num_bends]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            base_parameter.data.copy_(parameter.data)

    def init_linear(self):
        parameters = list(self.curve_architecture.parameters())
        for i in range(0, len(parameters), self.num_bends):
            weights = parameters[i:i+self.num_bends]
            for j in range(1, self.num_bends - 1):
                alpha = j * 1.0 / (self.num_bends - 1)
                weights[j].data.copy_(alpha * weights[-1].data + (1.0 - alpha) * weights[0].data)

    def weights(self, t):
        coeffs_t = self.curve.get_coeffs(t)
        weights = []
        for module in self.curve_modules:
            weights.extend([w for w in module.compute_weights_t(coeffs_t) if w is not None])
        return np.concatenate([w.detach().cpu().numpy().ravel() for w in weights])

    def _compute_l2(self):
        self.l2 = sum(module.l2 for module in self.curve_modules)

    def forward(self, input, t=None):
        if t is None:
            t = input.data.new(1).uniform_()
        coeffs_t = self.curve.get_coeffs(t)
        output = self.curve_architecture(input, coeffs_t)
        self._compute_l2()
        return output


def l2_regularizer(weight_decay):
    return lambda model: 0.5 * weight_decay * model.l2


if(__name__ == "__main__"):
    cs = CurveSystem(3,3,PolyChain)
    print(cs(1/3))
    print(cs(1/12))
    print(cs(1/12 + 1/24))
    print(cs(1/12 + 1/3))
    print(cs(1/12 + 2/3))

    
