import yz_template as yz
import pytorch_lightning as pl
import torch


class End2EndSystem(yz.TrainModule):
    def __init__(self):
        super(End2EndSystem, self).__init__()
        self.model1.unfreeze()
        self.model2.freeze()
    
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        optim_1 = torch.optim.Adam(self.model1.parameters(), lr=self.args.lr1)
        optim_2 = torch.optim.Adam(self.model2.parameters(), lr=self.args.lr2)
        return [optim_1, optim_2]

    @yz.args
    def set_args(self):
        self.parser.add_argument("--lr1", default=1e-6, type=float, help="Stage1 Learning rate for optimizer")
        self.parser.add_argument("--lr2", default=0, type=float, help="Stage2 Learning rate for optimizer")

    
End2EndModel = End2EndSystem()
trainer = pl.Trainer(gpus='%d' % End2EndModel.args.cuda)
trainer.fit(End2EndModel)