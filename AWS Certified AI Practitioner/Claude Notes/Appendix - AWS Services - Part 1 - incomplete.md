# AWS Services Study Guide
## Analytics, Cloud Financial Management, Compute, Containers & Database Services

---

## Analytics Services

### AWS Data Exchange
**Purpose**: Marketplace for finding, subscribing to, and using third-party data in the cloud.

**Key Features**:
- **Data Marketplace**: Browse and subscribe to data products from third-party providers
- **Data Delivery**: Automatic delivery of data updates to your S3 buckets
- **API Integration**: RESTful APIs for programmatic access
- **Billing Integration**: Usage-based billing through AWS

**Data Types Available**:
- Financial market data
- Weather and climate data
- Demographics and census data
- Healthcare and life sciences data
- Geospatial and mapping data

**AI/ML Use Cases**:
- Enriching training datasets with external data sources
- Real-time market data for financial ML models
- Weather data for demand forecasting models
- Demographics data for customer segmentation

**Pricing Model**: Pay-per-use or subscription-based, depending on data provider

**Integration**: Works seamlessly with S3, Lambda, SageMaker, and other AWS analytics services

---

### Amazon EMR (Elastic MapReduce)
**Purpose**: Managed big data platform for processing large datasets using open-source tools.

**Core Components**:
- **Master Node**: Coordinates cluster activities and task distribution
- **Core Nodes**: Run tasks and store data in HDFS
- **Task Nodes**: Run tasks only (optional, for additional compute)

**Supported Frameworks**:
- **Apache Spark**: Fast, general-purpose distributed computing
- **Apache Hadoop**: Distributed storage and processing framework
- **Apache Hive**: Data warehouse software for querying large datasets
- **Apache HBase**: NoSQL database for real-time read/write access
- **Presto**: Distributed SQL query engine
- **Apache Zeppelin**: Web-based notebook for interactive analytics

**Instance Types**:
- **On-Demand**: Standard pricing, pay for what you use
- **Reserved**: 1-3 year commitments with significant savings
- **Spot**: Up to 90% savings using spare EC2 capacity

**AI/ML Applications**:
- **Data Preprocessing**: Clean and transform large datasets for ML
- **Feature Engineering**: Extract and create features from raw data
- **Model Training**: Distributed training of ML models using Spark MLlib
- **Batch Inference**: Process large datasets for predictions

**Security Features**:
- **Encryption**: At-rest and in-transit encryption
- **IAM Integration**: Fine-grained access control
- **VPC Support**: Deploy in private subnets
- **Kerberos**: Authentication for Hadoop ecosystem

**Best Practices**:
- Use appropriate instance types for workload characteristics
- Leverage spot instances for cost optimization
- Monitor cluster utilization and auto-scaling
- Use S3 for persistent data storage

---

### AWS Glue
**Purpose**: Fully managed extract, transform, and load (ETL) service for preparing data for analytics.

**Key Components**:

#### Data Catalog
- **Metadata Repository**: Central repository for metadata about data sources
- **Schema Discovery**: Automatically discovers and catalogs data schemas
- **Table Definitions**: Stores table schemas, partition information, and data location
- **Integration**: Works with Athena, EMR, Redshift Spectrum, and SageMaker

#### ETL Jobs
- **Serverless**: No infrastructure to manage
- **Apache Spark**: Runs on managed Spark environment
- **Python/Scala**: Support for both programming languages
- **Visual ETL**: Drag-and-drop interface for creating ETL workflows

#### Crawlers
- **Auto-Discovery**: Automatically discover data sources and extract metadata
- **Schema Evolution**: Detect and handle schema changes over time
- **Scheduling**: Run on schedules or triggered by events
- **Supported Sources**: S3, RDS, Redshift, DynamoDB, and JDBC sources

#### DataBrew Integration
- **Visual Data Preparation**: GUI-based data cleaning and transformation
- **Recipe Management**: Reusable transformation recipes
- **Data Quality**: Built-in data quality checks and profiling

**AI/ML Use Cases**:
- **Data Pipeline Creation**: Build ETL pipelines to prepare training data
- **Feature Store**: Create and manage feature datasets for ML models
- **Data Quality**: Ensure data quality before feeding into ML workflows
- **Schema Management**: Handle evolving data schemas in ML pipelines

**Pricing**: Pay for resources used during ETL job execution and Data Catalog requests

**Integration**: Native integration with S3, Athena, QuickSight, SageMaker, and other AWS services

---

### AWS Glue DataBrew
**Purpose**: Visual data preparation tool for cleaning and normalizing data without writing code.

**Key Capabilities**:

#### Visual Interface
- **Point-and-Click**: No coding required for data transformations
- **Recipe Builder**: Create reusable transformation recipes
- **Data Preview**: See transformation results in real-time
- **Profile Insights**: Automated data quality and statistical insights

#### Data Transformation Functions**:
- **Data Cleaning**: Remove duplicates, handle missing values, standardize formats
- **Data Normalization**: Standardize data formats and values
- **Data Enrichment**: Add calculated columns and derived metrics
- **Data Filtering**: Filter rows and columns based on conditions

#### Data Sources
- **S3**: Primary data source for files
- **Data Catalog**: Use Glue Data Catalog tables
- **Redshift**: Connect to Redshift tables
- **RDS**: Connect to relational databases

**Profile Jobs**:
- **Data Quality Assessment**: Identify data quality issues
- **Statistical Analysis**: Generate descriptive statistics
- **Data Distribution**: Understand data patterns and distributions
- **Anomaly Detection**: Identify outliers and unusual patterns

**Recipe Jobs**:
- **Batch Processing**: Apply transformations to entire datasets
- **Incremental Processing**: Process only new or changed data
- **Scheduling**: Automated job execution
- **Monitoring**: Track job performance and success rates

**AI/ML Applications**:
- **Data Preparation**: Prepare raw data for ML model training
- **Feature Engineering**: Create new features from existing data
- **Data Quality**: Ensure high-quality data for better model performance
- **Exploratory Data Analysis**: Understand data characteristics before modeling

**Pricing**: Pay for profiling and recipe job execution time

---

### AWS Lake Formation
**Purpose**: Service to build, secure, and manage data lakes on AWS.

**Core Capabilities**:

#### Data Lake Creation
- **Centralized Setup**: Single service to set up data lakes
- **Data Ingestion**: Import data from various sources into S3
- **Metadata Management**: Automatically catalog and organize data
- **Schema Evolution**: Handle changing data schemas over time

#### Security and Governance
- **Fine-Grained Access Control**: Column and row-level security
- **Centralized Permissions**: Manage permissions across all data lake services
- **Audit Logging**: Track all data access and modifications
- **Data Classification**: Automatically classify sensitive data

#### Data Discovery and Access
- **Search Capabilities**: Find relevant datasets across the data lake
- **Self-Service Access**: Allow users to discover and access data independently
- **Query Integration**: Works with Athena, Redshift Spectrum, EMR, and QuickSight

**Key Features**:
- **Blueprints**: Pre-built workflows for common data ingestion patterns
- **Transactions**: ACID transactions for data consistency
- **Time Travel**: Query historical versions of data
- **Compaction**: Optimize storage and query performance

**Integration with Analytics Services**:
- **Amazon Athena**: Query data using SQL without moving it
- **Amazon Redshift**: Load and analyze data in data warehouse
- **Amazon EMR**: Process data using big data frameworks
- **Amazon QuickSight**: Create visualizations and dashboards

**AI/ML Benefits**:
- **Centralized Data Access**: Single point of access for all ML training data
- **Data Governance**: Ensure data quality and compliance for ML workflows
- **Feature Discovery**: Help data scientists find relevant features
- **Secure Data Sharing**: Share data across teams while maintaining security

**Pricing**: Pay for underlying AWS services used (S3, Glue, etc.) plus Lake Formation management costs

---

### Amazon OpenSearch Service
**Purpose**: Managed search and analytics engine for log analytics, real-time application monitoring, and clickstream analytics.

**Core Capabilities**:

#### Search and Analytics
- **Full-Text Search**: Advanced search capabilities with relevance scoring
- **Real-Time Analytics**: Analyze data as it's ingested
- **Aggregations**: Complex data aggregations and grouping
- **Geospatial Search**: Location-based search and analytics

#### Data Ingestion
- **Multiple Sources**: Ingest from CloudWatch, Kinesis, S3, and other sources
- **Real-Time Streaming**: Process data streams in real-time
- **Batch Processing**: Load large datasets for analysis
- **Data Transformation**: Transform data during ingestion

#### Visualization Tools
- **OpenSearch Dashboards**: Built-in visualization and dashboard creation
- **Kibana Compatibility**: Works with existing Kibana dashboards
- **Custom Visualizations**: Create custom charts and graphs
- **Real-Time Monitoring**: Monitor metrics and logs in real-time

**Cluster Architecture**:
- **Master Nodes**: Manage cluster state and metadata
- **Data Nodes**: Store data and execute queries
- **Ingest Nodes**: Process incoming data before indexing
- **Coordinating Nodes**: Route requests and merge results

**AI/ML Integration**:
- **Anomaly Detection**: Built-in ML for detecting anomalies in time-series data
- **k-NN Search**: k-nearest neighbor search for similarity matching
- **Learning to Rank**: ML-powered search result ranking
- **Performance Analyzer**: ML-driven performance insights

**Security Features**:
- **Fine-Grained Access Control**: Control access at index and field level
- **Encryption**: At-rest and in-transit encryption
- **VPC Support**: Deploy in private networks
- **SAML and Active Directory**: Enterprise authentication integration

**Use Cases**:
- **Log Analytics**: Centralized logging and analysis
- **Application Monitoring**: Real-time application performance monitoring
- **Security Analytics**: Security event analysis and threat detection
- **Business Intelligence**: Search-driven analytics and reporting

**Pricing**: Pay for compute instances, storage, and data transfer

---

### Amazon QuickSight
**Purpose**: Fast, cloud-powered business intelligence service for creating visualizations and dashboards.

**Key Features**:

#### Data Sources
- **AWS Services**: S3, Redshift, RDS, Athena, Aurora, DynamoDB
- **SaaS Applications**: Salesforce, ServiceNow, Jira, GitHub
- **Databases**: MySQL, PostgreSQL, SQL Server, Oracle, Teradata
- **Files**: Excel, CSV, JSON, Apache Parquet, Apache ORC

#### Visualization Types
- **Charts**: Bar, line, pie, scatter, heat maps, treemaps
- **Tables**: Pivot tables, data tables with conditional formatting
- **Maps**: Geospatial visualizations with location data
- **Custom Visuals**: Extend with custom visualization types

#### Advanced Analytics
- **Machine Learning Insights**: Automated insights using ML
- **Forecasting**: Time-series forecasting with seasonal patterns
- **Anomaly Detection**: Identify unusual patterns in data
- **Natural Language Queries**: Ask questions in plain English

#### SPICE Engine
- **In-Memory Processing**: Super-fast, Parallel, In-memory Calculation Engine
- **Columnar Storage**: Optimized for analytical queries
- **Automatic Scaling**: Scales to handle large datasets
- **Compression**: Efficient data compression for cost optimization

**Editions**:
- **Standard**: Basic BI capabilities with pay-per-session pricing
- **Enterprise**: Advanced features, hourly refresh, row-level security

**Security and Governance**:
- **Row-Level Security**: Control data access at the row level
- **Column-Level Security**: Hide sensitive columns from users
- **Active Directory Integration**: Enterprise authentication
- **Data Encryption**: Encrypt data at rest and in transit

**AI/ML Dashboard Integration**:
- **SageMaker Integration**: Visualize ML model results and metrics
- **Model Performance Monitoring**: Track model accuracy and drift
- **Feature Importance**: Visualize which features impact predictions
- **A/B Testing Results**: Compare model performance across experiments

**Pricing Models**:
- **Pay-per-Session**: Pay only when users access dashboards
- **Annual Subscription**: Fixed monthly cost per user
- **Embedded Analytics**: Custom pricing for embedding in applications

---

### Amazon Redshift
**Purpose**: Fast, fully managed data warehouse for analyzing large datasets using SQL.

**Architecture**:

#### Cluster Components
- **Leader Node**: Coordinates query execution and communicates with client applications
- **Compute Nodes**: Store data and execute queries in parallel
- **Node Slices**: Each compute node divided into slices for parallel processing

#### Storage Types
- **Dense Storage (DS2)**: HDD-based for large data warehouses
- **Dense Compute (DC2)**: SSD-based for high-performance workloads
- **Redshift Spectrum**: Query data directly in S3 without loading

**Performance Optimization**:

#### Columnar Storage
- **Column-Oriented**: Store data by columns rather than rows
- **Compression**: Automatic compression reduces storage and I/O
- **Zone Maps**: Automatically created to skip irrelevant blocks

#### Parallel Processing
- **Massively Parallel**: Distribute queries across all compute nodes
- **Node-to-Node Communication**: Efficient data movement between nodes
- **Automatic Query Optimization**: Cost-based query optimizer

#### Advanced Features
- **Materialized Views**: Pre-computed query results for faster access
- **Result Caching**: Cache frequently accessed query results
- **Workload Management**: Prioritize and allocate resources to queries

**Data Loading Methods**:
- **COPY Command**: Bulk load data from S3, DynamoDB, or remote hosts
- **INSERT Statements**: Load small amounts of data
- **AWS Data Pipeline**: Orchestrate complex data loading workflows
- **AWS Glue**: ETL service integration for data preparation

**Security Features**:
- **Encryption**: At-rest encryption using AWS KMS or HSM
- **Network Isolation**: VPC deployment for network security
- **IAM Integration**: Fine-grained access control
- **Audit Logging**: Track user activity and query execution

**AI/ML Integration**:
- **Amazon SageMaker**: Train and deploy ML models using Redshift data
- **Redshift ML**: Create, train, and deploy ML models using SQL
- **Feature Engineering**: Use SQL for feature creation and transformation
- **Model Inference**: Apply trained models to new data in Redshift

**Pricing Components**:
- **Compute**: Hourly rates based on node type and quantity
- **Storage**: Additional storage beyond included amount
- **Data Transfer**: Cross-region and internet data transfer
- **Backup Storage**: Automated and manual snapshot storage

---

## Cloud Financial Management

### AWS Budgets
**Purpose**: Set custom cost and usage budgets to track AWS spending and usage.

**Budget Types**:

#### Cost Budgets
- **Track Spending**: Monitor actual and forecasted costs
- **Budget Periods**: Monthly, quarterly, or annual tracking
- **Granularity**: Account-level, service-level, or tag-based budgets
- **Currency Support**: Multiple currencies supported

#### Usage Budgets
- **Resource Utilization**: Track usage of specific AWS services
- **Unit Tracking**: Monitor hours, requests, or other service-specific units
- **Service Coverage**: EC2, RDS, S3, Lambda, and other services
- **Reserved Instance Utilization**: Track RI usage efficiency

#### Reservation Budgets
- **RI Coverage**: Monitor Reserved Instance coverage percentage
- **RI Utilization**: Track how well RIs are being used
- **Savings Plans**: Monitor Savings Plans utilization and coverage

#### Savings Plans Budgets
- **Coverage Tracking**: Monitor Savings Plans coverage
- **Utilization Monitoring**: Track Savings Plans utilization rates
- **Cost Savings**: Measure actual savings achieved

**Alert Mechanisms**:
- **Email Notifications**: Send alerts to specified email addresses
- **SNS Integration**: Trigger SNS topics for custom notifications
- **Threshold Types**: Actual spend, forecasted spend, or percentage thresholds
- **Multiple Thresholds**: Set up to 5 alert thresholds per budget

**Advanced Features**:
- **Filtering**: Filter by service, linked account, tag, or other dimensions
- **Time-Based Budgets**: Create budgets for specific time periods
- **Budget Reports**: Detailed spending reports and analysis
- **API Access**: Programmatic budget management and monitoring

**AI/ML Cost Management**:
- **SageMaker Training**: Budget for training job costs
- **Inference Endpoints**: Monitor real-time endpoint costs
- **Data Storage**: Track S3 costs for training datasets
- **Compute Resources**: Monitor EC2 costs for ML workloads

**Best Practices**:
- Start with broad budgets and refine over time
- Use tags for granular cost tracking
- Set up multiple alert thresholds
- Regular review and adjustment of budget amounts

**Pricing**: First 2 budgets free, then $0.02 per budget per day

---

### AWS Cost Explorer
**Purpose**: Visualize, understand, and manage AWS costs and usage over time.

**Core Capabilities**:

#### Cost Visualization
- **Interactive Charts**: Customizable charts and graphs
- **Time Ranges**: View costs across different time periods
- **Granularity**: Daily, monthly, or yearly cost breakdowns
- **Service Breakdown**: Costs by AWS service
- **Account Breakdown**: Costs by linked accounts (for organizations)

#### Usage Analysis
- **Resource Utilization**: Analyze how resources are being used
- **Service Metrics**: Detailed usage metrics for each service
- **Trend Analysis**: Identify cost and usage trends over time
- **Comparative Analysis**: Compare costs across different periods

#### Filtering and Grouping
- **Dimension Filtering**: Filter by service, region, account, or tags
- **Group By Options**: Group costs by various dimensions
- **Tag-Based Analysis**: Analyze costs using resource tags
- **Custom Filters**: Create complex filtering criteria

**Rightsizing Recommendations**:
- **EC2 Rightsizing**: Identify over-provisioned EC2 instances
- **Cost Savings**: Potential savings from rightsizing
- **Performance Impact**: Assess impact of size changes
- **Implementation Guidance**: Step-by-step rightsizing instructions

**Reserved Instance Recommendations**:
- **Purchase Recommendations**: Suggest RI purchases for cost savings
- **Utilization Analysis**: Track existing RI utilization
- **Coverage Reports**: Monitor RI coverage across resources
- **ROI Calculations**: Calculate return on investment for RIs

**Savings Plans Recommendations**:
- **Commitment Recommendations**: Suggest optimal commitment levels
- **Savings Potential**: Calculate potential savings
- **Coverage Analysis**: Monitor Savings Plans coverage
- **Utilization Tracking**: Track Savings Plans utilization rates

**Reporting Features**:
- **Custom Reports**: Create and save custom cost reports
- **Scheduled Reports**: Automatically generated and delivered reports
- **CSV Export**: Export data for external analysis
- **API Access**: Programmatic access to cost and usage data

**AI/ML Cost Analysis**:
- **SageMaker Costs**: Analyze training, inference, and storage costs
- **Data Transfer Costs**: Monitor costs for data movement
- **Compute Optimization**: Identify opportunities for cost optimization
- **Resource Tagging**: Use tags to track ML project costs

**Best Practices**:
- Set up consistent tagging strategy for better cost attribution
- Regular review of recommendations and implementation
- Use grouping and filtering for detailed analysis
- Monitor cost trends and anomalies

**Pricing**: Basic Cost Explorer is free; advanced features have additional costs

---

## Compute Services

### Amazon EC2 (Elastic Compute Cloud)
**Purpose**: Resizable compute capacity in the cloud with complete control over computing resources.

**Instance Categories**:

#### General Purpose
- **A1**: ARM-based processors, cost-effective for scale-out workloads
- **M5/M5a/M5n**: Balanced compute, memory, and networking
- **M6i**: Latest generation with improved price-performance
- **T3/T4g**: Burstable performance with baseline CPU credits

#### Compute Optimized
- **C5/C5n**: High-performance processors for compute-intensive applications
- **C6i**: Latest generation with enhanced networking
- **C6g**: ARM-based Graviton2 processors for improved price-performance

#### Memory Optimized
- **R5/R5a/R5n**: High memory-to-vCPU ratio for memory-intensive applications
- **R6g**: ARM-based with large memory capacity
- **X1e**: Highest memory per vCPU ratio for in-memory databases
- **z1d**: High-frequency processors with NVMe SSD storage

#### Storage Optimized
- **I3/I3en**: NVMe SSD-backed instance storage for high I/O performance
- **D2**: Dense HDD storage for distributed file systems
- **H1**: High disk throughput with 16 TB of HDD storage per instance

#### Accelerated Computing
- **P3/P4**: GPU instances for machine learning and high-performance computing
- **G4**: GPU instances for graphics-intensive applications
- **F1**: FPGA instances for hardware acceleration
- **Inf1**: Machine learning inference with AWS Inferentia chips

**Purchasing Options**:

#### On-Demand Instances
- **Pay-as-you-go**: No upfront costs or long-term commitments
- **Flexibility**: Start and stop instances as needed
- **Use Cases**: Variable workloads, development/testing, short-term needs

#### Reserved Instances
- **Commitment**: 1 or 3-year terms with significant savings (up to 75%)
- **Types**: Standard (highest savings), Convertible (flexibility), Scheduled
- **Payment Options**: All upfront, partial upfront, or no upfront

#### Spot Instances
- **Cost Savings**: Up to 90% savings using spare EC2 capacity
- **Interruption**: Can be terminated with 2-minute notice
- **Use Cases**: Fault-tolerant, flexible applications, batch processing

#### Dedicated Hosts
- **Physical Servers**: Dedicated physical servers for compliance requirements
- **License Optimization**: Bring your own licenses (BYOL)
- **Compliance**: Meet regulatory requirements for dedicated hardware

**Storage Options**:

#### Elastic Block Store (EBS)
- **General Purpose SSD (gp3)**: Balanced price and performance
- **Provisioned IOPS SSD (io2)**: High-performance for critical workloads
- **Throughput Optimized HDD (st1)**: Low-cost HDD for throughput-intensive workloads
- **Cold HDD (sc1)**: Lowest cost HDD for infrequently accessed data

#### Instance Store
- **Temporary Storage**: Storage physically attached to host computer
- **High Performance**: Very high I/O performance
- **Ephemeral**: Data lost when instance stops or terminates

**Networking Features**:
- **Enhanced Networking**: SR-IOV for high-bandwidth, low-latency networking
- **Placement Groups**: Control instance placement for optimal performance
- **Elastic Network Interfaces**: Multiple network interfaces per instance
- **IPv6 Support**: Dual-stack IPv4 and IPv6 addressing

**AI/ML Specific Considerations**:

#### GPU Instances (P3, P4, G4)
- **Deep Learning**: Optimized for training and inference workloads
- **CUDA Support**: NVIDIA GPU computing platform
- **Multi-GPU**: Multiple GPUs per instance for distributed training
- **Memory**: High memory capacity for large models and datasets

#### CPU Instances for ML
- **Compute Optimized**: C5 instances for CPU-intensive ML workloads
- **Memory Optimized**: R5 instances for large dataset processing
- **Inference Optimization**: M5 instances for cost-effective inference

**Security Features**:
- **Security Groups**: Virtual firewalls controlling inbound/outbound traffic
- **Key Pairs**: Secure SSH access using public-key cryptography
- **IAM Roles**: Secure access to AWS services without hardcoded credentials
- **Encryption**: EBS encryption and in-transit encryption options

**Monitoring and Management**:
- **CloudWatch**: Detailed monitoring and alerting
- **Systems Manager**: Patch management and remote access
- **Auto Scaling**: Automatically adjust capacity based on demand
- **Load Balancing**: Distribute traffic across multiple instances

**Best Practices**:
- Choose appropriate instance types based on workload characteristics
- Use Auto Scaling for variable workloads
- Implement proper security groups and IAM policies
- Monitor performance and costs regularly
- Use appropriate storage types for different data access patterns

**Pricing Factors**:
- Instance type and size
- Purchasing option (On-Demand, Reserved, Spot)
- Operating system
- Region and Availability Zone
- Data transfer
- Additional features (monitoring, storage, etc.)

---

## Container Services

### Amazon Elastic Container Service (Amazon ECS)
**Purpose**: Fully managed container orchestration service for running Docker containers at scale.

**Core Components**:

#### Clusters
- **Logical Grouping**: Group of EC2 instances or Fargate tasks
- **Resource Management**: Manages compute resources for containers
- **Scaling**: Automatic scaling based on resource needs
- **Multi-AZ**: Deploy across multiple Availability Zones for high availability

#### Task Definitions
- **Container Blueprint**: JSON document describing container configuration
- **Container Specifications**: CPU, memory, port mappings, environment variables
- **Networking Mode**: Bridge, host, awsvpc, or none
- **Storage**: Volume mounts and storage configuration

#### Services
- **Desired State**: Maintain specified number of running tasks
- **Load Balancing**: Integration with Application Load Balancer and Network Load Balancer
- **Service Discovery**: Automatic service registration and discovery
- **Rolling Updates**: Zero-downtime deployments with rolling updates

#### Tasks
- **Running Containers**: One or more containers running together
- **Placement**: Automatic placement across cluster resources
- **Health Monitoring**: Automatic restart of unhealthy tasks
- **Resource Allocation**: CPU and memory allocation per task

**Launch Types**:

#### EC2 Launch Type
- **EC2 Instances**: Run containers on managed EC2 instances
- **Instance Management**: You manage EC2 instances in the cluster
- **Cost Control**: More granular control over compute costs
- **Customization**: Custom AMIs and instance configurations
- **Use Cases**: Long-running applications, cost optimization, custom requirements

#### Fargate Launch Type
- **Serverless**: No EC2 instances to manage
- **Task-Level Isolation**: Each task runs in its own kernel runtime environment
- **Automatic Scaling**: Scales individual tasks based on CPU and memory requirements
- **Pay-per-Task**: Pay only for the resources your tasks consume
- **Use Cases**: Microservices, batch jobs, simplicity over cost optimization

**Networking**:
- **VPC Integration**: Deploy containers in VPC with security groups
- **Service Mesh**: Support for AWS App Mesh for microservices communication
- **Load Balancing**: Integration with Elastic Load Balancing
- **Service Discovery**: Cloud Map integration for service discovery

**Security Features**:
- **IAM Roles**: Task-level IAM roles for secure access to AWS services
- **Secrets Management**: Integration with AWS Secrets Manager and Systems Manager Parameter Store
- **Network Isolation**: VPC and security group isolation
- **Image Scanning**: ECR image vulnerability scanning

**AI/ML Use Cases**:
- **Model Serving**: Deploy ML models as containerized microservices
- **Batch Processing**: Run ML training jobs using containers
- **Data Pipeline**: Containerized data processing workflows
- **Multi-Model Serving**: Deploy multiple models with different resource requirements

**Integration with AI/ML Services**:
- **SageMaker**: Use ECS for custom model serving endpoints
- **Batch Processing**: Container-based ML training pipelines
- **Real-time Inference**: Scalable model serving with load balancing
- **Data Processing**: ETL jobs using containerized applications

**Monitoring and Logging**:
- **CloudWatch**: Container and service-level monitoring
- **X-Ray**: Distributed tracing for containerized applications
- **Container Insights**: Detailed container performance metrics
- **Log Drivers**: Multiple log drivers including CloudWatch Logs

**Best Practices**:
- Use Fargate for simplicity, EC2 for cost optimization
- Implement proper health checks for containers
- Use service discovery for microservices communication
- Implement proper security with IAM roles and secrets management
- Monitor container performance and resource utilization

**Pricing**:
- **EC2 Launch Type**: Pay for EC2 instances and EBS volumes
- **Fargate Launch Type**: Pay for vCPU and memory resources consumed by tasks

---

### Amazon Elastic Kubernetes Service (Amazon EKS)
**Purpose**: Managed Kubernetes service for running Kubernetes applications without needing to install and operate Kubernetes clusters.

**Core Components**:

#### Control Plane
- **Managed Masters**: AWS manages Kubernetes master nodes
- **High Availability**: Multi-AZ deployment of master nodes
- **Patching**: Automatic security patching and updates
- **API Server**: Kubernetes API server with AWS authentication integration

#### Worker Nodes
- **Node Groups**: Managed groups of EC2 instances running kubelet
- **Fargate**: Serverless option for running pods
- **Spot Instances**: Cost optimization using Spot instances in node groups
- **Auto Scaling**: Automatic scaling of worker nodes based on demand

#### Networking
- **VPC Integration**: Native VPC networking for pods
- **CNI Plugin**: AWS VPC CNI plugin for pod networking
- **Load Balancing**: Integration with AWS Load Balancers
- **Ingress Controllers**: Support for various ingress controllers

**Node Types**:

#### Managed Node Groups
- **Simplified Management**: AWS manages EC2 instances lifecycle
- **Auto Scaling**: Built-in cluster autoscaler support
- **Rolling Updates**: Automatic rolling updates for worker nodes
- **Instance Types**: Support for various EC2 instance types including GPU instances

#### Self-Managed Nodes
- **Full Control**: Complete control over EC2 instances
- **Custom Configuration**: Custom AMIs and instance configurations
- **Cost Optimization**: Fine-grained cost control and optimization options
- **Advanced Networking**: Custom networking configurations

#### Fargate
- **Serverless Pods**: Run pods without managing EC2 instances
- **Pod-Level Isolation**: Each pod runs in its own compute environment
- **Automatic Scaling**: Scales pods based on resource requirements
- **Pay-per-Pod**: Pay only for the resources your pods consume

**Kubernetes Features**:
- **Latest Versions**: Support for latest Kubernetes versions
- **Native APIs**: Full access to Kubernetes APIs and features
- **Helm**: Native support for Helm package manager
- **Operators**: Support for Kubernetes operators
- **Custom Resources**: Full support for custom resource definitions

**Security Features**:
- **IAM Integration**: Kubernetes RBAC with AWS IAM
- **Pod Security**: Pod security policies and security contexts
- **Network Policies**: Kubernetes network policies for micro-segmentation
- **Secrets Management**: Integration with AWS Secrets Manager
- **Image Scanning**: ECR integration with vulnerability scanning

**AI/ML Capabilities**:

#### Machine Learning Workloads
- **GPU Support**: P3, P4, and G4 instances for ML training and inference
- **Kubeflow**: Support for Kubeflow ML pipelines
- **Distributed Training**: Multi-node distributed training using MPI and Horovod
- **Jupyter Notebooks**: JupyterHub deployment for data science workflows

#### ML Operators
- **TensorFlow Operator**: Distributed TensorFlow training
- **PyTorch Operator**: Distributed PyTorch training
- **MPI Operator**: Message Passing Interface for distributed computing
- **Argo Workflows**: ML pipeline orchestration

#### Integration with AWS AI Services
- **SageMaker**: Use EKS for custom training and inference
- **Batch Processing**: GPU-accelerated batch ML jobs
- **Model Serving**: Scalable model serving with Kubernetes services
- **Data Processing**: Spark on Kubernetes for big data processing

**Add-ons and Extensions**:
- **AWS Load Balancer Controller**: Advanced load balancing features
- **EBS CSI Driver**: Persistent storage using EBS volumes
- **EFS CSI Driver**: Shared storage using EFS
- **Cluster Autoscaler**: Automatic node scaling based on pod demand
- **AWS App Mesh**: Service mesh for microservices

**Monitoring and Observability**:
- **CloudWatch Container Insights**: Detailed container and cluster metrics
- **Prometheus**: Native Prometheus support for monitoring
- **Grafana**: Integration with Grafana for visualization
- **X-Ray**: Distributed tracing for applications
- **Fluentd**: Log aggregation and forwarding

**Best Practices**:
- Use managed node groups for simplicity
- Implement proper RBAC and security policies
- Use appropriate instance types for workloads (GPU for ML)
- Monitor cluster and application performance
- Implement proper resource requests and limits
- Use namespaces for resource isolation

**Pricing Components**:
- **Control Plane**: Fixed hourly charge per cluster
- **Worker Nodes**: EC2 instance costs (On-Demand, Reserved, or Spot)
- **Fargate**: Pay for vCPU and memory consumed by pods
- **Data Transfer**: Standard AWS data transfer charges

---

## Database Services

### Amazon DocumentDB (with MongoDB compatibility)
**Purpose**: Fully managed document database