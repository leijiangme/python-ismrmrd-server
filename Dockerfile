FROM kspacekelvin/fire-python:latest

# Copy the new file from the host into the image
# COPY <all> <the> <things> <last-arg-is-destination>/
# Note: the trailing / is required when copying from multiple sources.

COPY CSzeropadding.py server.py /opt/code/python-ismrmrd-server/
