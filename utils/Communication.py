import torch

def communication(opt, server_model, models, client_weights):
    print(opt.mode)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    server_model = server_model.to(device)
    for i, model in enumerate(models):
        models[i] = model.to(device)
    with torch.no_grad():
        # aggregate params
        if opt.mode.lower() == 'yang_geo':
            for key in server_model.state_dict().keys():
##                if 'weight_fed' not in key and 'MLP' not in key :
#                if 'keys' not in key:
                # if 'geo_model' not in key and  'tconv' not in key:
                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                for client_idx in range(len(client_weights)):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)

                if 'tconv' not in key:
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

        else:
            print('No Person')
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    
    server_model = server_model.cpu()
    for i, model in enumerate(models):
        models[i] = model.cpu()
    torch.cuda.empty_cache()
    
    return server_model, models