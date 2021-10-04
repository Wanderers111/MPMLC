import torch


def ranges_slices(batch):
    """

    :param batch:
    :return:
    """
    Ns = batch.bincount()
    indices = Ns.cumsums(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = (
        torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
    )
    slices = (1 + torch.arange(len(Ns))).int.to(batch.device)
    return ranges, slices


def diagonal_ranges(batch_x, batch_y):
    # If the batch is not defined, the range of the D_ij is None.
    if batch_x is None and batch_y is None:
        return None
    # If the batch_y is not defined, making the y = x as the unsupervised learning.
    elif batch_y is None:
        batch_y = batch_x
    return
