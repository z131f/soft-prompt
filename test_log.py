import logging
import os

# 获取logger实例，或者根logger（根据您的具体需求，这里使用您之前的方法__name__）
# 如果您想确保所有Trainer的日志也通过此logger，最好使用root_logger = logging.getLogger()
logger_to_configure = logging.getLogger(__name__) # 假设您仍想使用模块名作为logger名称
# 或者使用 root_logger = logging.getLogger() 来捕获所有日志

# 设置最低日志级别为INFO
logger_to_configure.setLevel(logging.INFO)

# 关键步骤：在每次配置前，移除所有现有的 handlers
for handler in logger_to_configure.handlers[:]: # 遍历handlers的副本，避免在迭代时修改列表
    logger_to_configure.removeHandler(handler)
    handler.close() # 关闭 handler 释放资源，尤其是FileHandler会占用文件句柄

# 1. 创建一个 StreamHandler (用于控制台输出)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # 设置控制台输出的最低级别

# 2. 创建一个 FileHandler (用于文件输出)
log_file_path = "app.log" # 定义日志文件名称
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True) # 如果logs目录不存在，则创建
file_handler = logging.FileHandler(os.path.join(log_dir, log_file_path))
file_handler.setLevel(logging.INFO) # 设置文件输出的最低级别

# 创建一个 Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 为两个handler设置Formatter
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将两个handler添加到logger
logger_to_configure.addHandler(console_handler)
logger_to_configure.addHandler(file_handler)
logger = logger_to_configure # 确保使用配置后的logger

logger.info("Logger has been configured successfully.")