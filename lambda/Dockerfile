FROM public.ecr.aws/lambda/python:3.11

RUN yum -y install gcc gcc-c++ make && yum clean all

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["app.lambda_handler"]