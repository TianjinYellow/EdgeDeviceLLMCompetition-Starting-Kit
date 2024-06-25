import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = '1'
os.environ["MKL_THREADING_LAYER"] = '1'

#os.environ['TRANSFORMERS_CACHE'] = '/scratch-share/transformers'
os.environ['HF_DATASETS_CACHE'] = '/scratch-shared/'
os.environ['HF_TOKENIZERS_CACHE'] = '/scratch-shared/tokenizes'
os.environ['HF_HOME'] = '/scratch-shared/HF_HOME'
os.environ['HF_METRICS_CACHE'] = '/scratch-shared/metrics'
os.environ['HF_MODULES_CACHE'] = '/scratch-shared/modules'

from opencompass.cli.main import main

if __name__ == '__main__':
    main()
