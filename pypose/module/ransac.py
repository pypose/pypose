import torch
from torch import nn

class RANSAC(nn.Module):
    def __init__(self, method = 'ransac'):
        super().__init__()
        self.method = method
        assert method in ["ransac", "ransac_adapt"], "method not supported"

    @staticmethod
    def ransac_adapt(data:dict, fit_model, check_model, confidence=0.999, num_of_select=8, threshold=1):
        r"""
        The algorithm identifies inliers and outliers.

        Args:
            data (``dict``): Data in custom dictionary form, can be used for custom models.
            fit_model: The function used to fit the data.
            check_model: The funciton used to check the fitted model.
            confidence (``float``, optional): The probabilty of obtaining a model fitted by all selected data that are inliers.
            num_of_select(``int``, optional)" The least number of data to fit the model.
            threshold (``float``, optional): The maximum tolerable error threshold.

        Returns:
            (``Mbest: torch.Tensor, mask_inliers: torch.Tensor``):
            ``Mbest``: The matrix with the most inliers.

            ``mask_inliers``: The index of the inliers in the corresponding points set. The shape is (..., 1).

        Example:
            >>> pts1 = torch.tensor([[ 40, 368],[ 63, 224], [ 85, 447], [108, 151], [ 77, 319], [411, 371],[335, 183],[ 42, 151],[144, 440],
            ...                      [407, 403],[208, 153], [216, 288], [156, 318], [104, 319], [ 72, 287], [373, 180], [383, 200], [184, 306],
            ...                      [362, 185], [211, 463], [385, 211], [ 80,  81],[ 78, 195], [398, 223]]).type(torch.FloatTensor)
            >>> pts2 = torch.tensor([[ 16.0156, 377.1875],[ 45.9062, 229.5703],[ 59.3750, 459.8750], [ 94.4766, 155.7422], [ 56.5469, 326.3945], [409.6777, 378.8906],
            ...                      [327.7773, 185.5117], [ 27.0625, 156.4453], [121.6250, 452.5000], [402.8477, 412.9609], [196.8281, 156.6094], [202.9375, 293.7734],
            ...                      [139.6562, 324.9727],[ 84.5312, 326.2695], [ 52.7656, 293.5781],[367.6094, 182.1094], [379.1543, 202.5859], [169.1406, 312.4688],
            ...                      [356.0312, 187.2891],[191.6719, 476.9531],[381.6484, 213.8477],[ 91.6328,  92.5391],[ 62.7266, 200.2344],[396.4111, 226.1543]])
            >>> pts1 = pp.cart2homo(pts1)
            >>> pts2 = pp.cart2homo(pts2)
            >>> from epipolar import *
            >>> def fit_model(data,samples):
            ...     coordinates1 = data['coordinates_1'][samples[:,:]]
            ...     coordinates2 = data['coordinates_2'][samples[:,:]]
            ...     return eight_pts_alg(coordinates1, coordinates2)
            >>> def check_model(data, M):
            ...     coordinates1 = data['coordinates_1']
            ...     coordinates2 = data['coordinates_2']
            ...     return compute_error(coordinates1, coordinates2, M)
            >>> ransac = RANSAC('ransac_adapt')
            >>> data = {'coordinates_1': pts1, 'coordinates_2': pts2, 'highest_int': len(pts1)}
            >>> num_of_select = 18
            >>> F, mask = ransac(data,fit_model,check_model,0.999,num_of_select,threshold = 0.5)
            >>> F
            tensor([[ 4.6117e-07, -9.0610e-05,  2.7522e-02],
                    [ 9.4227e-05,  3.9505e-06, -8.7046e-02],
                    [-2.9752e-02,  8.4636e-02,  1.0000e+00]])
        """
        assert isinstance(confidence, float) == True, "confidence has to be a float number"
        assert 0 <= confidence <= 1, "confidence should be within [0, 1]!"

        max_num_of_inliers = num_of_select
        h_int = data['highest_int']
        # k = 200
        k = 20000
        iters = 0
        while iters < k:

            samples = torch.randperm(h_int)[None, :num_of_select]
            # run fit model
            M = fit_model(data, samples)
            # check the data with M, get the err and mask for inliers
            err = check_model(data, M)

            mask = torch.argwhere(err <= threshold)

            if len(mask) > max_num_of_inliers:

                max_num_of_inliers = len(mask)
                mask_inliers = mask[:,1]
                t = len(mask)/h_int
                k = torch.log(torch.tensor(1 - confidence))/torch.log(torch.tensor(1 - t ** num_of_select))

                # refine the M with all inliers
                M = fit_model(data,torch.unsqueeze(mask[:,1],dim = 0))
                Mbest = M

            iters = iters + 1

        return Mbest, mask_inliers

    @staticmethod
    def ransac(data: dict, fit_model, check_model, iterations = 1000, num_of_select = 8, threshold = 1):
        r"""
        The algorithm identifies inliers and outliers.

        Args:
            data (``dict``): Data in custom dictionary form, can be used for custom models.
            fit_model: The function used to fit the data.
            check_model: The funciton used to check the fitted model.
            iterations (``int``, optional): The maximum number of iterations.
            num_of_select(``int``, optional)" The least number of data to fit the model.
            threshold (``float``, optional): The maximum tolerable error threshold.

        Returns:
            (``Mbest: torch.Tensor, mask_inliers: torch.Tensor``):
            ``Mbest``: The matrix with the most inliers.

            ``mask_inliers``: The index of the inliers in the corresponding points set. The shape is (..., 1).

        Example:
            >>> pts1 = torch.tensor([[ 40, 368],[ 63, 224], [ 85, 447], [108, 151], [ 77, 319], [411, 371],[335, 183],[ 42, 151],[144, 440],
            ...                      [407, 403],[208, 153], [216, 288], [156, 318], [104, 319], [ 72, 287], [373, 180], [383, 200], [184, 306],
            ...                      [362, 185], [211, 463], [385, 211], [ 80,  81],[ 78, 195], [398, 223]]).type(torch.FloatTensor)
            >>> pts2 = torch.tensor([[ 16.0156, 377.1875],[ 45.9062, 229.5703],[ 59.3750, 459.8750], [ 94.4766, 155.7422], [ 56.5469, 326.3945], [409.6777, 378.8906],
            ...                      [327.7773, 185.5117], [ 27.0625, 156.4453], [121.6250, 452.5000], [402.8477, 412.9609], [196.8281, 156.6094], [202.9375, 293.7734],
            ...                      [139.6562, 324.9727],[ 84.5312, 326.2695], [ 52.7656, 293.5781],[367.6094, 182.1094], [379.1543, 202.5859], [169.1406, 312.4688],
            ...                      [356.0312, 187.2891],[191.6719, 476.9531],[381.6484, 213.8477],[ 91.6328,  92.5391],[ 62.7266, 200.2344],[396.4111, 226.1543]])
            >>> pts1 = pp.cart2homo(pts1)
            >>> pts2 = pp.cart2homo(pts2)
            >>> from epipolar import *
            >>> def fit_model(data,samples):
            ...     coordinates1 = data['coordinates_1'][samples[:,:]]
            ...     coordinates2 = data['coordinates_2'][samples[:,:]]
            ...     return eight_pts_alg(coordinates1, coordinates2)
            >>> def check_model(data, M):
            ...     coordinates1 = data['coordinates_1']
            ...     coordinates2 = data['coordinates_2']
            ...     return compute_error(coordinates1, coordinates2, M)
            >>> ransac = RANSAC('ransac')
            >>> data = {'coordinates_1': pts1, 'coordinates_2': pts2, 'highest_int': len(pts1)}
            >>> num_of_select = 18
            >>> F, mask = ransac(data,fit_model,check_model,2000,num_of_select = 8 ,threshold = 0.5)
            >>> F
            tensor([[ 4.6117e-07, -9.0610e-05,  2.7522e-02],
                    [ 9.4227e-05,  3.9505e-06, -8.7046e-02],
                    [-2.9752e-02,  8.4636e-02,  1.0000e+00]])
        """
        assert isinstance(iterations, int) == True, "The parameter 'terminate' should be an positive integer!"

        samples = torch.randint(0, data['highest_int'], (iterations, num_of_select))

        # unique check
        unique_seq =[]
        for i in range(iterations):
            if len(torch.unique(samples[i])) == num_of_select:
                unique_seq.append(i)

        if len(unique_seq) == 0:
            raise Exception("Unique check alert! Please reduce the number of select or increase the number of iterations!")

        samples = samples[torch.tensor(unique_seq)]

        M = fit_model(data, samples) # (..., n, 3)

        err = check_model(data, M)

        mask_lens = torch.sum(err <= threshold,dim = -1)
        mask_len_ratio = mask_lens/ mask_lens.max()

        err_inliers = torch.where(err<= threshold, err, 0)
        err_inlier_means = (torch.sqrt(err_inliers).sum(dim = -1) + 1e-8)/(mask_lens + 1e-8)
        err_ratio = err_inlier_means/err_inlier_means.max()

        score = 0.5 * mask_len_ratio + 0.5 * ( 1 - err_ratio)

        max_score_seq = score.argmax()
        mask_inliers = torch.argwhere(err[max_score_seq] <= threshold)

        if len(mask_inliers) >= num_of_select:
            # refine the M with all inliers
            Mbest = fit_model(data, mask_inliers.transpose(-2,-1))
        else:
             raise Exception("Could not find a model due to insufficient inliers!")

        return Mbest, mask_inliers

    def forward(self, data, fit_model, check_model, terminate, num_of_select = 8, threshold=1):

        if self.method == 'ransac':
            return self.ransac(data, fit_model, check_model, terminate, num_of_select, threshold)
        elif self.method == 'ransac_adapt':
            return self.ransac_adapt(data, fit_model, check_model, terminate, num_of_select, threshold)
        else:
            raise NotImplementedError("Method not implemented")
