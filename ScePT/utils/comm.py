import torch
import torch.distributed as dist


def all_gather(data):
    """Run all_gather on arbitrary picklable data (not necessarily tensors)
    
    Parameters
    ----------
    data: any picklable object
    
    Returns
    --------
    list[data]
        List of data gathered from each rank
    """
    world_size = dist.get_world_size()

    if world_size == 1:
        return [data]

    # Serialize to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # Obtain Tensor size of each rank
    local_size = torch.IntTensor([tensor.numel()]).to("cuda")
    size_list = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # Receive Tensor from all ranks
    # We pad the tensor because torch all_gather does not support gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size, )).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size, )).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
