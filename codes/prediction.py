import os
import torch
import numpy as np
import U_net_Model
import scipy.ndimage
from medpy import metric
import myconfig
import qU_net_Model
import config

threshold = 2
batch_size = 2

def generate_quantize_model(model):
    config_file = 'config.yaml'
    model_type = 'UNET'
    kconfig = config.Configuration(model_type, config_file)
    kconfig = kconfig.config_dict
    qmodel=qU_net_Model.qUNet(in_channels=5,out_channels=2, config=kconfig)
    for (layer, qlayer) in zip(model.named_parameters(), qmodel.named_parameters()):
        for tensor, target_tensor in zip(layer[1], qlayer[1]):
          tensor.copy_(target_tensor.detach().clone())
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    qmodel = qmodel.to(device)
    return qmodel

def load_model(parameters_dict_fn):
    model=U_net_Model.UNet(in_channels=5,out_channels=2)
    model_dict = torch.load(parameters_dict_fn)["state_dict"]
    model.load_state_dict(model_dict)
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)
    qmodel = generate_quantize_model(model)
    return qmodel

def arr_at_each_location(image,direction,loc):
    arr=[]
    x_size,y_size,z_size=image.shape
    if direction=="X":
        assert 0<=loc<image.shape[0]
        arr=[
            image[loc-5,:,:] if loc-5>=0 else np.zeros((y_size,z_size)),
            image[loc-2,:,:] if loc-2>=0 else np.zeros((y_size,z_size)),
            image[loc,:,:],
            image[loc+2,:,:] if loc+2<x_size else np.zeros((y_size,z_size)),
            image[loc+5,:,:] if loc+5<x_size else np.zeros((y_size,z_size))
        ]
        arr=np.stack(arr,0)[np.newaxis,:,:,:]
    elif direction=="Y":
        assert 0<=loc<image.shape[1]
        arr=[
            image[:,loc-5,:] if loc-5>=0 else np.zeros((x_size,z_size)),
            image[:,loc-2,:] if loc-2>=0 else np.zeros((x_size,z_size)),
            image[:,loc,:],
            image[:,loc+2,:] if loc+2<y_size else np.zeros((x_size,z_size)),
            image[:,loc+5,:] if loc+5<y_size else np.zeros((x_size,z_size))
        ]
        arr=np.stack(arr,0)[np.newaxis,:,:,:]
    elif direction=="Z":
        assert 0<=loc<image.shape[2]
        arr=[
            image[:,:,loc-5] if loc-5>=0 else np.zeros((x_size,y_size)),
            image[:,:,loc-2] if loc-2>=0 else np.zeros((x_size,y_size)),
            image[:,:,loc],
            image[:,:,loc+2] if loc+2<z_size else np.zeros((x_size,y_size)),
            image[:,:,loc+5] if loc+5<z_size else np.zeros((x_size,y_size)),
        ]
        arr=np.stack(arr,0)[np.newaxis,:,:,:]
    else:
        assert False
    return arr.astype(np.float32)

def get_prediction(unet, direction, image,batch_size):

    x_size,y_size,z_size=image.shape
    method = torch.nn.Softmax(dim=1)
    unet.eval()
    with torch.no_grad():
        if direction=="X":
            tumor_distribution_prediction=np.zeros([x_size,y_size,z_size])
            for i in range(0,x_size,batch_size):
                idx_range=range(i,min(i+batch_size,x_size))
                arr=np.concatenate([arr_at_each_location(image,direction,m) for m in idx_range],axis=0)
                batch_input=torch.from_numpy(arr).cuda()
                predict=unet(batch_input)
                predict=method(predict)
                predict=predict.cpu().numpy()[:,1,:,:]
                tumor_distribution_prediction[idx_range,:,:]=predict
        elif direction=="Y":
            tumor_distribution_prediction=np.zeros([x_size,y_size,z_size])
            for i in range(0,y_size,batch_size):
                idx_range=range(i,min(i+batch_size,y_size))
                arr=np.concatenate([arr_at_each_location(image,direction,m) for m in idx_range],axis=0)
                batch_input=torch.from_numpy(arr).cuda()
                predict=unet(batch_input)
                predict=method(predict)
                predict=predict.cpu().numpy()[:,1,:,:]
                tumor_distribution_prediction[:,idx_range,:]=predict.transpose([1,0,2])
        elif direction=="Z":
            tumor_distribution_prediction=np.zeros([x_size,y_size,z_size])
            for i in range(0,z_size,batch_size):
                idx_range=range(i,min(i+batch_size,z_size))
                arr=np.concatenate([arr_at_each_location(image,direction,m) for m in idx_range],axis=0)
                batch_input=torch.from_numpy(arr).cuda()
                predict=unet(batch_input)
                predict=method(predict)
                predict=predict.cpu().numpy()[:,1,:,:]
                tumor_distribution_prediction[:,:,idx_range]=predict.transpose([1,2,0])
        else:
            assert False
    
    return tumor_distribution_prediction

def final_prediction(image,best_model_fns,threshold=2,filter_predictions=True,lung_mask=None,direction="all",batch_size=16):
    if direction=="all":
        unet=load_model(best_model_fns['X'])
        x_prediction = get_prediction(unet, 'X', image,batch_size)
        load_model(best_model_fns['Y'])
        y_prediction = get_prediction(unet, 'Y', image,batch_size)
        load_model(best_model_fns['Z'])
        z_prediction = get_prediction(unet, 'Z', image,batch_size)
        pred = np.array(x_prediction+y_prediction+z_prediction > threshold, 'float32')
    elif direction=="X":
        unet=load_model(best_model_fns['X'])
        pred = get_prediction(unet, 'X', image,batch_size)>threshold
    elif direction=="Y":
        unet=load_model(best_model_fns['Y'])
        pred = get_prediction(unet, 'Y', image,batch_size)>threshold       
    elif direction=="Z":
        unet=load_model(best_model_fns['Z'])
        pred = get_prediction(unet, 'Z', image,batch_size)>threshold   
    else:
        assert False
    if filter_predictions:
        pred=scipy.ndimage.maximum_filter(pred,3)
    if type(lung_mask)==np.ndarray:
        pred=pred*lung_mask
    return pred

def predict_lung_mask(data_array, args):
    # if lung_mask is an (512, 512, 512) np array, then it will use it to discard FP outside lung_mask.
    # warning, lung_mask must be rescaled
    # save both the normalized and un-normalized mask

    best_model_fns = {
        'X': os.path.join(args.model_lung_dir, "best_model-X.pth"),
        'Y': os.path.join(args.model_lung_dir, "best_model-Y.pth"),
        'Z': os.path.join(args.model_lung_dir, "best_model-Z.pth")
    }
    prediction = final_prediction(data_array, best_model_fns, threshold=threshold, lung_mask=None, batch_size=batch_size)
    return prediction

def predict_one_scan(patient_id, args):
    # if lung_mask is an (512, 512, 512) np array, then it will use it to discard FP outside lung_mask.
    # warning, lung_mask must be rescaled
    # save both the normalized and un-normalized mask
    data_path = args.test_data_dir + patient_id + '/' + patient_id + '.npz'
    label_path = args.test_data_dir + patient_id + '/' + patient_id + '_label.npz'
    data_array = np.load(data_path)['array']  # the data_array are normalized, with shape (512, 512, 512)
    label_array = np.load(label_path)['array']

    best_model_fns = {
        'X': os.path.join(args.model_infection_dir, "best_model-X.pth"),
        'Y': os.path.join(args.model_infection_dir, "best_model-Y.pth"),
        'Z': os.path.join(args.model_infection_dir, "best_model-Z.pth")
    }

    lung_mask = predict_lung_mask(data_array, args)

    prediction = final_prediction(data_array, best_model_fns, threshold=threshold, lung_mask=lung_mask, batch_size=batch_size)

    dice = metric.dc(prediction, label_array)

    prediction = final_prediction(data_array, best_model_fns, threshold=threshold, lung_mask=lung_mask, batch_size=batch_size)
    return prediction, dice

if __name__ == "__main__":
    # Once we have the pretrained model, we can validate our result
    args = myconfig.get_config()
    patient_ids = os.listdir(args.test_data_dir)
    for patient_id in patient_ids:
        print(f'processing patient: {patient_id}')
        prediction, dice = predict_one_scan(patient_id, args)
        print(f'The prediction dice is: {dice}')
        np.savez_compressed(os.path.join('./result',patient_id+'_prediction'),array=prediction)
    pass
