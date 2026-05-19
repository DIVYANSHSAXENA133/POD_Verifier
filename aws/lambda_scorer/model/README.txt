Canonical trained checkpoint committed to Git:

  aws/lambda_scorer/model/best.pt

Replace this file with your trained MultiHeadEfficientNet weights, then add and push:

  cp /path/to/your/trained/best.pt aws/lambda_scorer/model/best.pt
  git add aws/lambda_scorer/model/best.pt
  git commit -m "Add trained POD checkpoint"
  git push

Only this .pt path is un-ignored by the repo .gitignore.

Optional — dummy initializer for CI (not for production inference):

  python3 aws/scripts/create_init_checkpoint.py
