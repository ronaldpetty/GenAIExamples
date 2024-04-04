FROM vault.habana.ai/gaudi-docker/1.15.1/ubuntu22.04/habanalabs/pytorch-installer-2.2.0:latest
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /home/user
COPY requirements.txt /home/user/
RUN pip install -r requirements.txt
# RUN pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.13.0
# RUN pip install --upgrade-strategy eager optimum[habana]
# # enable password-less ssh
# RUN ssh-keygen -t rsa -f /root/.ssh/id_rsa -P '' && \
#     cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys
# RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# CMD ["sh", "-c", "service ssh restart; bash"]