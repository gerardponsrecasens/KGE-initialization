from .utils import *
from .model.model_process import *
from .model.model_process_incDE import *

class Trainer():
    def __init__(self, args, kg, model, optimizer):
        self.args = args
        self.kg = kg  
        self.model = model
        self.optimizer = optimizer
        self.logger = args.logger
        if args.lifelong_name == 'incDE':
            self.train_processor = TrainBatchProcessorincDE(args, kg)
            self.valid_processor = DevBatchProcessorincDE(args, kg)
        else:
            self.train_processor = TrainBatchProcessor(args, kg)
            self.valid_processor = DevBatchProcessor(args, kg)


    def run_epoch(self):
        self.args.valid = True
        loss = self.train_processor.process_epoch(self.model, self.optimizer)
        res = self.valid_processor.process_epoch(self.model)
        self.args.valid = False
        return loss, res


