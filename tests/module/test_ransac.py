import torch, pypose as pp


class TestRANSAC:

    def test_ransac_adapt(self):

        pts1 = torch.tensor([[ 40, 368], [ 63, 224], [ 85, 447], [108, 151], [ 77, 319],
                             [411, 371], [335, 183], [ 42, 151], [144, 440], [407, 403],
                             [208, 153], [216, 288], [156, 318], [104, 319], [ 72, 287],
                             [373, 180], [383, 200], [184, 306], [362, 185], [211, 463],
                             [385, 211], [ 80,  81], [ 78, 195], [398, 223]],
                             dtype=torch.float32)

        pts2 = torch.tensor([[ 16.0156, 377.1875], [ 45.9062, 229.5703], [ 59.3750, 459.8750],
                             [ 94.4766, 155.7422], [ 56.5469, 326.3945], [409.6777, 378.8906],
                             [327.7773, 185.5117], [ 27.0625, 156.4453], [121.6250, 452.5000],
                             [402.8477, 412.9609], [196.8281, 156.6094], [202.9375, 293.7734],
                             [139.6562, 324.9727], [ 84.5312, 326.2695], [ 52.7656, 293.5781],
                             [367.6094, 182.1094], [379.1543, 202.5859], [169.1406, 312.4688],
                             [356.0312, 187.2891], [191.6719, 476.9531], [381.6484, 213.8477],
                             [ 91.6328,  92.5391], [ 62.7266, 200.2344], [396.4111, 226.1543]],
                             dtype=torch.float32)

        pts1 = pp.cart2homo(pts1)
        pts2 = pp.cart2homo(pts2)

        def fit_model(data, samples):
            coordinates1 = data['coordinates_1'][samples[:,:]]
            coordinates2 = data['coordinates_2'][samples[:,:]]
            return pp.module.eight_pts_alg(coordinates1, coordinates2)

        def check_model(data, M):
            coordinates1 = data['coordinates_1']
            coordinates2 = data['coordinates_2']
            return pp.module.compute_error(coordinates1, coordinates2, M)

        ransac = pp.module.RANSAC('ransac_adapt')
        data = {'coordinates_1': pts1, 'coordinates_2': pts2, 'highest_int': len(pts1)}
        num_of_select = 18
        F, mask = ransac(data, fit_model, check_model, 0.999, num_of_select, threshold=0.5)
        print(F)

    def test_ransac(self):

        pts1 = torch.tensor([[ 40, 368], [ 63, 224], [ 85, 447], [108, 151], [ 77, 319],
                             [411, 371], [335, 183], [ 42, 151], [144, 440], [407, 403],
                             [208, 153], [216, 288], [156, 318], [104, 319], [ 72, 287],
                             [373, 180], [383, 200], [184, 306], [362, 185], [211, 463],
                             [385, 211], [ 80,  81], [ 78, 195], [398, 223]],
                             dtype=torch.float32)

        pts2 = torch.tensor([[ 16.0156, 377.1875], [ 45.9062, 229.5703], [ 59.3750, 459.8750],
                             [ 94.4766, 155.7422], [ 56.5469, 326.3945], [409.6777, 378.8906],
                             [327.7773, 185.5117], [ 27.0625, 156.4453], [121.6250, 452.5000],
                             [402.8477, 412.9609], [196.8281, 156.6094], [202.9375, 293.7734],
                             [139.6562, 324.9727], [ 84.5312, 326.2695], [ 52.7656, 293.5781],
                             [367.6094, 182.1094], [379.1543, 202.5859], [169.1406, 312.4688],
                             [356.0312, 187.2891], [191.6719, 476.9531], [381.6484, 213.8477],
                             [ 91.6328,  92.5391], [ 62.7266, 200.2344], [396.4111, 226.1543]],
                             dtype=torch.float32)

        pts1 = pp.cart2homo(pts1)
        pts2 = pp.cart2homo(pts2)

        def fit_model(data,samples):
            coordinates1 = data['coordinates_1'][samples[:,:]]
            coordinates2 = data['coordinates_2'][samples[:,:]]
            return pp.module.eight_pts_alg(coordinates1, coordinates2)

        def check_model(data, M):
            coordinates1 = data['coordinates_1']
            coordinates2 = data['coordinates_2']
            return pp.module.compute_error(coordinates1, coordinates2, M)

        ransac = pp.module.RANSAC('ransac')
        data = {'coordinates_1': pts1, 'coordinates_2': pts2, 'highest_int': len(pts1)}

        F, mask = ransac(data, fit_model, check_model, 2000, num_of_select=8 ,threshold=0.5)

        print(F)


if __name__ == "__main__":
    test = TestRANSAC()
    test.test_ransac()
    test.test_ransac_adapt()
