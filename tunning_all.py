import yz_template as yz
import pytorch_lightning as pl
import torch
import os


class End2EndSystem(yz.TrainModule):
    def __init__(self):
        super(End2EndSystem, self).__init__()
        self.model1.load_from_checkpoint('/home/yinzi/data3/stage1_retrained')
        self.model1.unfreeze()
        self.model2.load_from_checkpoint('/home/yinzi/data3/stage2_12_23')
        self.model2.unfreeze()
    
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        optim_1 = torch.optim.Adam(self.model1.parameters(), lr=self.args.lr1)
        optim_eyebrows = torch.optim.Adam(self.model2.model[0].parameters(), lr=self.args.lr_brows)
        optim_eyes = torch.optim.Adam(self.model2.model[1].parameters(), lr=self.args.lr_eyes)
        optim_nose = torch.optim.Adam(self.model2.model[2].parameters(), lr=self.args.lr_nose)
        optim_mouth = torch.optim.Adam(self.model2.model[3].parameters(), lr=self.args.lr_mouth)

        return [optim_1, optim_eyes, optim_eyebrows, optim_nose, optim_mouth]

    @yz.args
    def set_args(self):
        self.parser.add_argument("--epochs", default=25, type=int, help="Epochs")
        self.parser.add_argument("--lr1", default=1e-6, type=float, help="Stage1 Learning rate for optimizer")
        self.parser.add_argument("--lr_eyes", default=1e-6, type=float, help="eyes Learning rate for optimizer")
        self.parser.add_argument("--lr_brows", default=1e-6, type=float, help="eyebrows Learning rate for optimizer")
        self.parser.add_argument("--lr_nose", default=1e-6, type=float, help="nose Learning rate for optimizer")
        self.parser.add_argument("--lr_mouth", default=1e-6, type=float, help="mouth Learning rate for optimizer")
        self.parser.add_argument("--optim1_step", default=2, type=int, help="update stage1 opt every ? steps")
        self.parser.add_argument("--optim_eyse_step", default=2, type=int, help="update eyes opt every ? steps")
        self.parser.add_argument("--optim1_brows_step", default=2, type=int, help="update brows opt every ? steps")
        self.parser.add_argument("--optim1_nose_step", default=2, type=int, help="update nose opt every ? steps")
        self.parser.add_argument("--optim1_mouth_step", default=2, type=int, help="update mouth opt every ? steps")

    def on_save_checkpoint(self, checkpoint):
        pass

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        if optimizer_i == 0:
            if batch_nb % self.args.optim1_step == 0 :
                optimizer.step()
                optimizer.zero_grad()

        if optimizer_i == 1:
            if batch_nb % self.args.eyebrows_step == 0 :
                optimizer.step()
                optimizer.zero_grad()    

        if optimizer_i == 2:
            if batch_nb % self.args.optim_eyes_step == 0 :
                optimizer.step()
                optimizer.zero_grad()    

        if optimizer_i == 3:
            if batch_nb % self.args.optim_nose_step4 == 0 :
                optimizer.step()
                optimizer.zero_grad()     

        if optimizer_i == 4:
            if batch_nb % self.args.optim_mouth_step4 == 0 :
                optimizer.step()
                optimizer.zero_grad()


End2EndModel = End2EndSystem()
trainer = pl.Trainer(gpus='%d' % End2EndModel.args.cuda, min_nb_epochs=1, max_nb_epochs=End2EndModel.args.epochs)
trainer.fit(End2EndModel)
