import os
from configuration import *
from trainer import Trainer, tune

if __name__ == '__main__':
    config = get_parameters()

    # Ensure model_name is set
    if not config.model_name:
        config.model_name = 'Transformer'  # Default to Transformer if not specified

    print(f'Model name: {config.model_name}')  # Print to verify the model name
    print('-- init trainer')
    trainer = Trainer(config)
    if config.mode == 'train':
        print('-- start training')
        trainer.train()
    if config.mode == 'test':
        print('-- start testing')
        trainer.evaluate_test('test', 0, trainer.model, trainer.test_loader)
    if config.mode == 'tune':
        trainer.tune()
    if config.mode == 'pred':
        RAW_INPUT = {
            't1': '',
            't2': ''
        }
        trainer.predict(trainer.model, RAW_INPUT)
