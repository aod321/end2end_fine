import yz_template as yz
import pytorch_lightning as pl


class End2EndSystem(yz.TrainModule):
    def __init__(self):
        super(End2EndSystem, self).__init__()
        self.model1.freeze()
        self.model2.unfreeze()
    
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        optim_1 = torch.optim.Adam(self.model1.parameters(), lr=0)
        optim_2 = torch.optim.Adam(self.model2.parameters(), lr=1e-6)
        return [optim_1, optim_2]

    @args
    def set_args(self):
        self.parser.add_argument("--lr1", default=0, type=float, help="Stage1 Learning rate for optimizer")
        self.parser.add_argument("--lr2", default=1e-6, type=float, help="Stage2 Learning rate for optimizer")

    
End2EndModel = End2EndSystem()
trainer = pl.Trainer(gpus='%d' % End2EndModel.args.cuda)
trainer.fit(End2EndModel)