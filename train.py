from utils import parser
from tools import processor

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    args = parser.Parser().args
    print(args)
    process = processor.Processor(args)
    process.start()


if __name__ == '__main__':
    print(f'\n\n+-------------------------------- [Start training] --------------------------------+\n\n')
    main()

