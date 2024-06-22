import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = '1'
os.environ["MKL_THREADING_LAYER"] = '1'

#os.environ['TRANSFORMERS_CACHE'] = '/scratch-share/HTJ/transformers'
os.environ['HF_DATASETS_CACHE'] = '/scratch-shared/HTJ/'
os.environ['HF_TOKENIZERS_CACHE'] = '/scratch-shared/HTJ/tokenizes'
os.environ['HF_HOME'] = '/scratch-shared/HTJ/HF_HOME'
os.environ['HF_METRICS_CACHE'] = '/scratch-shared/HTJ/metrics'
os.environ['HF_MODULES_CACHE'] = '/scratch-shared/HTJ/modules'

from opencompass.cli.main import main

if __name__ == '__main__':
    main()
