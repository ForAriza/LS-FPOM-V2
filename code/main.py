import os
from LS_FPOM_v2 import LsFPOM_v2
from pre_process import PreProcess
from post_process import PostProcess
class Execute:
    def __init__(self, num_dataset):
        self.num_dataset = num_dataset
        self.path_parameter = os.path.dirname(os.path.dirname(__file__)) + r'/data/parameter.txt'
    def get_parameter(self):
        file = open(self.path_parameter)
        parameters = file.readlines()
        return eval(parameters[0])
    def exec(self):
        print('Loading Parameter')
        parameter = self.get_parameter()
        list_sample = parameter[0]
        list_transform = parameter[1]
        list_index = parameter[2]
        list_epoch = parameter[3]
        list_interval = parameter[4]
        list_weight_decay = parameter[5]
        list_learning_rate = parameter[6]
        list_local = parameter[7]
        list_point = parameter[8]
        list_peek = parameter[9]
        list_valley = parameter[10]
        print('Successfully Load Parameter')

        print('Initializing Preprocess Pipeline')
        pre_process = PreProcess(
            sample=list_sample[self.num_dataset],
            index=list_index[self.num_dataset],
            transform=list_transform[2]
        )
        print('Successfully Initialize Preprocess Pipeline')
        print('Executing Preprocess Pipeline')
        pre_process.exec()
        print('Successfully Execute Preprocessor Pipeline')
        del pre_process
        print('Initializing LsFPOMer_v2 Pipeline')
        LsFPOMer = LsFPOM_v2(
            sample=list_sample[self.num_dataset],
            epoch=list_epoch[self.num_dataset],
            interval=list_interval[self.num_dataset],
            weight_decay=list_weight_decay[self.num_dataset],
            learning_rate=list_learning_rate[self.num_dataset]
        )
        print('Successfully Initialize LsFPOMer_v2 Pipeline')
        LsFPOMer.exec()
        print('Successfully Execute LsFPOMer_v2 Pipeline')
        del LsFPOMer
        print('Initializing Postprocess Pipeline')
        post_process = PostProcess(
            index_sample=self.num_dataset
        )
        print('Successfully Initialize PostProcess Pipeline')
        print('OUC_Orient Mapping')
        post_process.draw_ouc_alpha_mapping(
            list_local=list_local[self.num_dataset]
        )
        print('OUC Mapping')
        post_process.draw_ouc_mapping(
            list_local=list_local[self.num_dataset]
        )
        print('Contrast Evaluating')
        post_process.get_line_data(
            p1=list_point[self.num_dataset][0],
            p2=list_point[self.num_dataset][1],
            list_local=list_local[self.num_dataset],
            index_peek=list_peek[self.num_dataset],
            index_valley=list_valley[self.num_dataset]
        )
        print('Successfully Execute Postprocessor Pipeline')
        del post_process

if __name__ == '__main__':
    Executor = Execute(0)
    Executor.exec()