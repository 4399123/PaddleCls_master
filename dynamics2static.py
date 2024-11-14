import paddle
from paddle.vision.models import mobilenet_v1
from paddle.static import InputSpec
from paddle.jit import to_static

#动态图模型路径

dynamics_model_path='./dynamics_model/model.pdparams'

net=mobilenet_v1(num_classes=2,pretrained=False)
layer_state_dict = paddle.load(dynamics_model_path)

net.set_state_dict(layer_state_dict)
net.eval()

net = to_static(net,input_spec=[InputSpec(shape=[None, 3, 224, 224], name='input')])

paddle.jit.save(net,'./static_model/model')

# net2=paddle.jit.load('./static_model/model')