import os
from datetime import datetime, timedelta, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

# from langchain.agents import load_tools
from langchain.agents.middleware import wrap_tool_call
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from rag_chain import load_qa_chain


class AwsCpuCheckInput(BaseModel):
    """Input for AWS CPU check queries."""

    instance_id: str = Field(description="EC2 instance ID (format: i-xxxxxxxxx)")
    hours: int = Field(
        default=1, description="Time range for the query (hours), default is 1 hour"
    )


# 载入环境变量
load_dotenv()

# === 初始化数据库 ===
db_uri = os.getenv("DB_URI")
db = SQLDatabase.from_uri(db_uri)

# === AWS客户端初始化 ===
cloudwatch_client = None


def get_cloudwatch_client():
    """获取CloudWatch客户端，实现懒加载和重用"""
    global cloudwatch_client
    if cloudwatch_client is None:
        # 从环境变量或 .env 文件读取 AWS 配置
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")  # 默认使用 us-east-1

        # 检查必需的凭证
        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError(
                "AWS 凭证未配置。请在 .env 文件中设置:\n"
                "AWS_ACCESS_KEY_ID=your_access_key\n"
                "AWS_SECRET_ACCESS_KEY=your_secret_key\n"
                "AWS_DEFAULT_REGION=your_region"
            )

        cloudwatch_client = boto3.client(
            "cloudwatch",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )
    return cloudwatch_client


# === 定义 AWS 工具 ===
@tool("get_ec2_cpu_usage", return_direct=True, args_schema=AwsCpuCheckInput)
def get_ec2_cpu_usage(instance_id: str, hours: int = 1):
    """
    获取指定EC2实例的平均CPU使用率

    Args:
        instance_id: EC2实例ID (格式: i-xxxxxxxxx)
        hours: 查询时间范围（小时），默认1小时
    """
    # 输入验证
    if not instance_id or not instance_id.startswith("i-"):
        return f"无效的实例ID格式: {instance_id}。实例ID应以'i-'开头"

    if hours <= 0 or hours > 24:
        return "时间范围应在1-24小时之间"

    try:
        cloudwatch = get_cloudwatch_client()

        # 使用UTC时区
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)

        response = cloudwatch.get_metric_statistics(
            Namespace="AWS/EC2",
            MetricName="CPUUtilization",
            Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,  # 5分钟间隔
            Statistics=["Average"],
        )

        datapoints = response.get("Datapoints", [])
        if not datapoints:
            return f"实例 {instance_id} 在过去 {hours} 小时内暂无CPU数据"

        # 按时间排序数据点，确保一致性
        datapoints.sort(key=lambda x: x["Timestamp"])

        # 计算平均值
        avg_cpu = sum(point["Average"] for point in datapoints) / len(datapoints)

        # 获取最新数据点的时间
        latest_time = max(point["Timestamp"] for point in datapoints)

        return (
            f"实例 {instance_id} 在过去 {hours} 小时内的平均CPU使用率为 {avg_cpu:.2f}%\n"
            f"数据点数量: {len(datapoints)}\n"
            f"最新数据时间: {latest_time.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "InvalidInstanceID.NotFound":
            return f"实例 {instance_id} 不存在"
        elif error_code == "UnauthorizedOperation":
            return "权限不足，无法访问CloudWatch指标"
        else:
            return f"AWS API错误: {e.response['Error']['Message']}"
    except BotoCoreError as e:
        return f"AWS连接错误: {str(e)}"
    except Exception as e:
        return f"获取CPU使用率时发生错误: {str(e)}"


# RAG 文档检索工具
qa_chain = load_qa_chain(embedding_mode="ollama")


@tool("qa_chain", return_direct=True)
def qa_chain_tool(query: str):
    """
    Search and answer questions from the documentation knowledge base.
    Use this tool when users ask questions about documentation, guides, or knowledge.

    Args:
        query: The question to search for in the knowledge base
    """
    try:
        result = qa_chain.invoke(query)
        return result
    except Exception as e:
        return f"搜索文档时出错: {str(e)}"


# db_tool = load_tools(["sql-database"], db=db)[0]


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"],
        )
