-- ============================================================================
-- PostgreSQL Database Schema for Wafer Defect Classification Platform
-- ============================================================================
-- 
-- Tables:
-- 1. users: User authentication and authorization
-- 2. wafer_maps: Raw wafer map metadata and storage references
-- 3. annotations: Ground truth segmentation annotations (COCO format)
-- 4. training_jobs: ML training job tracking and configuration
-- 5. defect_embeddings: Feature embeddings for active learning
-- 6. active_learning_queue: Query strategy and sample selection
-- 7. annotation_metrics: Inter-annotator agreement and quality tracking
-- 8. inference_logs: Production inference monitoring and audit trail
--
-- ============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- 1. Users Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('admin', 'annotator', 'engineer', 'viewer')),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    
    -- Annotation metrics (for annotators)
    total_annotations INTEGER DEFAULT 0,
    average_annotation_time_seconds FLOAT,
    inter_annotator_agreement FLOAT,
    
    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);

-- ============================================================================
-- 2. Wafer Maps Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS wafer_maps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    wafer_id VARCHAR(255) UNIQUE NOT NULL,
    lot_id VARCHAR(100),
    die_size_x FLOAT,
    die_size_y FLOAT,
    
    -- Storage
    image_path VARCHAR(500) NOT NULL,  -- S3/MinIO path
    image_format VARCHAR(10) DEFAULT 'png',
    image_size_bytes BIGINT,
    image_width INTEGER,
    image_height INTEGER,
    
    -- Processing status
    status VARCHAR(50) DEFAULT 'uploaded' CHECK (
        status IN ('uploaded', 'queued', 'annotated', 'training', 'validated', 'archived')
    ),
    
    -- Timestamps
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    metadata JSONB,  -- Flexible field for additional data
    
    -- Foreign keys
    uploaded_by UUID REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX idx_wafer_maps_wafer_id ON wafer_maps(wafer_id);
CREATE INDEX idx_wafer_maps_lot_id ON wafer_maps(lot_id);
CREATE INDEX idx_wafer_maps_status ON wafer_maps(status);
CREATE INDEX idx_wafer_maps_uploaded_at ON wafer_maps(uploaded_at);
CREATE INDEX idx_wafer_maps_metadata ON wafer_maps USING GIN(metadata);

-- ============================================================================
-- 3. Annotations Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS annotations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    wafer_map_id UUID REFERENCES wafer_maps(id) ON DELETE CASCADE,
    
    -- Annotation data (COCO JSON format)
    segmentation JSONB NOT NULL,  -- Polygon coordinates
    category_id INTEGER NOT NULL CHECK (category_id BETWEEN 0 AND 7),  -- 8 defect classes
    category_name VARCHAR(50),
    
    -- Bounding box
    bbox_x FLOAT,
    bbox_y FLOAT,
    bbox_width FLOAT,
    bbox_height FLOAT,
    area FLOAT,
    
    -- Quality metrics
    confidence FLOAT CHECK (confidence BETWEEN 0 AND 1),
    is_verified BOOLEAN DEFAULT FALSE,
    is_active_learning_sample BOOLEAN DEFAULT FALSE,
    
    -- Annotator info
    annotated_by UUID REFERENCES users(id) ON DELETE SET NULL,
    annotation_time_seconds INTEGER,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    verified_at TIMESTAMP WITH TIME ZONE,
    verified_by UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Version control
    version INTEGER DEFAULT 1,
    parent_annotation_id UUID REFERENCES annotations(id) ON DELETE SET NULL
);

CREATE INDEX idx_annotations_wafer_map_id ON annotations(wafer_map_id);
CREATE INDEX idx_annotations_category_id ON annotations(category_id);
CREATE INDEX idx_annotations_annotated_by ON annotations(annotated_by);
CREATE INDEX idx_annotations_is_verified ON annotations(is_verified);
CREATE INDEX idx_annotations_created_at ON annotations(created_at);
CREATE INDEX idx_annotations_segmentation ON annotations USING GIN(segmentation);

-- ============================================================================
-- 4. Training Jobs Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS training_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_name VARCHAR(255) NOT NULL,
    
    -- Job configuration
    model_architecture VARCHAR(100) DEFAULT 'resnet50_unet',
    num_epochs INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    learning_rate FLOAT NOT NULL,
    loss_function VARCHAR(50) DEFAULT 'combined',
    
    -- Training mode
    training_mode VARCHAR(50) DEFAULT 'supervised' CHECK (
        training_mode IN ('supervised', 'active_learning', 'semi_supervised', 'fine_tuning')
    ),
    
    -- Dataset info
    total_samples INTEGER,
    train_samples INTEGER,
    val_samples INTEGER,
    test_samples INTEGER,
    num_labeled_samples INTEGER,
    num_unlabeled_samples INTEGER,
    
    -- Status
    status VARCHAR(50) DEFAULT 'pending' CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'cancelled')
    ),
    
    -- Results
    best_val_iou FLOAT,
    best_val_dice FLOAT,
    final_train_loss FLOAT,
    final_val_loss FLOAT,
    
    -- Kubernetes job info
    k8s_job_name VARCHAR(255),
    k8s_namespace VARCHAR(100) DEFAULT 'ml-training',
    
    -- MLflow tracking
    mlflow_run_id VARCHAR(255),
    mlflow_experiment_id VARCHAR(255),
    
    -- Storage
    model_checkpoint_path VARCHAR(500),
    onnx_model_path VARCHAR(500),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- User tracking
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Hyperparameters (flexible JSON storage)
    hyperparameters JSONB,
    
    -- Logs and errors
    error_message TEXT,
    training_logs TEXT
);

CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_training_jobs_training_mode ON training_jobs(training_mode);
CREATE INDEX idx_training_jobs_created_at ON training_jobs(created_at);
CREATE INDEX idx_training_jobs_mlflow_run_id ON training_jobs(mlflow_run_id);

-- ============================================================================
-- 5. Defect Embeddings Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS defect_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    wafer_map_id UUID REFERENCES wafer_maps(id) ON DELETE CASCADE,
    training_job_id UUID REFERENCES training_jobs(id) ON DELETE SET NULL,
    
    -- Embedding vector (2048-dim from ResNet-50 stage4)
    embedding FLOAT[] NOT NULL,
    embedding_dim INTEGER NOT NULL DEFAULT 2048,
    
    -- Uncertainty metrics for active learning
    prediction_entropy FLOAT,  -- H(p) = -Σ p_i log(p_i)
    bald_score FLOAT,          -- Bayesian Active Learning by Disagreement
    variation_ratio FLOAT,
    
    -- Predicted class distribution
    class_probabilities FLOAT[] CHECK (array_length(class_probabilities, 1) = 8),
    predicted_class INTEGER CHECK (predicted_class BETWEEN 0 AND 7),
    prediction_confidence FLOAT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_embedding_dim CHECK (array_length(embedding, 1) = embedding_dim)
);

CREATE INDEX idx_defect_embeddings_wafer_map_id ON defect_embeddings(wafer_map_id);
CREATE INDEX idx_defect_embeddings_training_job_id ON defect_embeddings(training_job_id);
CREATE INDEX idx_defect_embeddings_prediction_entropy ON defect_embeddings(prediction_entropy);
CREATE INDEX idx_defect_embeddings_predicted_class ON defect_embeddings(predicted_class);

-- ============================================================================
-- 6. Active Learning Queue Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS active_learning_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    wafer_map_id UUID REFERENCES wafer_maps(id) ON DELETE CASCADE,
    training_job_id UUID REFERENCES training_jobs(id) ON DELETE CASCADE,
    
    -- Query strategy
    query_strategy VARCHAR(50) DEFAULT 'hybrid' CHECK (
        query_strategy IN ('uncertainty', 'diversity', 'hybrid', 'random')
    ),
    
    -- Scores for selection
    uncertainty_score FLOAT,   -- From entropy, BALD, etc.
    diversity_score FLOAT,     -- From CoreSet, k-center greedy, etc.
    combined_score FLOAT,      -- λ_uncertainty * uncertainty + λ_diversity * diversity
    
    -- Priority and status
    priority INTEGER DEFAULT 0,  -- Higher = more important
    status VARCHAR(50) DEFAULT 'pending' CHECK (
        status IN ('pending', 'assigned', 'annotated', 'skipped', 'rejected')
    ),
    
    -- Assignment
    assigned_to UUID REFERENCES users(id) ON DELETE SET NULL,
    assigned_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    query_iteration INTEGER,  -- Active learning iteration number
    batch_id UUID,            -- Group samples from same query
    
    UNIQUE(wafer_map_id, training_job_id)
);

CREATE INDEX idx_active_learning_queue_status ON active_learning_queue(status);
CREATE INDEX idx_active_learning_queue_combined_score ON active_learning_queue(combined_score DESC);
CREATE INDEX idx_active_learning_queue_assigned_to ON active_learning_queue(assigned_to);
CREATE INDEX idx_active_learning_queue_query_iteration ON active_learning_queue(query_iteration);

-- ============================================================================
-- 7. Annotation Metrics Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS annotation_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Pairwise inter-annotator agreement
    annotator1_id UUID REFERENCES users(id) ON DELETE CASCADE,
    annotator2_id UUID REFERENCES users(id) ON DELETE CASCADE,
    wafer_map_id UUID REFERENCES wafer_maps(id) ON DELETE CASCADE,
    
    -- Agreement metrics
    iou_agreement FLOAT CHECK (iou_agreement BETWEEN 0 AND 1),
    dice_agreement FLOAT CHECK (dice_agreement BETWEEN 0 AND 1),
    pixel_agreement FLOAT CHECK (pixel_agreement BETWEEN 0 AND 1),
    
    -- Annotation comparison
    annotation1_id UUID REFERENCES annotations(id) ON DELETE CASCADE,
    annotation2_id UUID REFERENCES annotations(id) ON DELETE CASCADE,
    
    -- Timestamps
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT different_annotators CHECK (annotator1_id != annotator2_id)
);

CREATE INDEX idx_annotation_metrics_annotator1 ON annotation_metrics(annotator1_id);
CREATE INDEX idx_annotation_metrics_annotator2 ON annotation_metrics(annotator2_id);
CREATE INDEX idx_annotation_metrics_wafer_map ON annotation_metrics(wafer_map_id);

-- ============================================================================
-- 8. Inference Logs Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS inference_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    wafer_map_id UUID REFERENCES wafer_maps(id) ON DELETE CASCADE,
    
    -- Model info
    model_version VARCHAR(100),
    model_path VARCHAR(500),
    
    -- Prediction results
    predicted_mask_path VARCHAR(500),  -- S3/MinIO path to saved mask
    class_distribution JSONB,          -- Count per defect class
    total_defect_area FLOAT,
    confidence_scores FLOAT[] CHECK (array_length(confidence_scores, 1) = 8),
    
    -- Performance metrics
    inference_time_ms FLOAT,
    preprocessing_time_ms FLOAT,
    postprocessing_time_ms FLOAT,
    
    -- Request metadata
    request_id VARCHAR(255),
    client_ip VARCHAR(45),
    api_endpoint VARCHAR(255),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Status
    status VARCHAR(50) DEFAULT 'success' CHECK (status IN ('success', 'error')),
    error_message TEXT
);

CREATE INDEX idx_inference_logs_wafer_map_id ON inference_logs(wafer_map_id);
CREATE INDEX idx_inference_logs_model_version ON inference_logs(model_version);
CREATE INDEX idx_inference_logs_created_at ON inference_logs(created_at);
CREATE INDEX idx_inference_logs_status ON inference_logs(status);

-- ============================================================================
-- Views for Analytics
-- ============================================================================

-- View: Annotation progress by user
CREATE OR REPLACE VIEW annotation_progress_by_user AS
SELECT 
    u.id AS user_id,
    u.username,
    u.role,
    COUNT(a.id) AS total_annotations,
    COUNT(CASE WHEN a.is_verified THEN 1 END) AS verified_annotations,
    AVG(a.annotation_time_seconds) AS avg_annotation_time,
    MAX(a.created_at) AS last_annotation_date
FROM users u
LEFT JOIN annotations a ON u.id = a.annotated_by
WHERE u.role = 'annotator' AND u.is_active = TRUE
GROUP BY u.id, u.username, u.role;

-- View: Training job summary
CREATE OR REPLACE VIEW training_job_summary AS
SELECT 
    training_mode,
    status,
    COUNT(*) AS job_count,
    AVG(best_val_iou) AS avg_best_iou,
    AVG(best_val_dice) AS avg_best_dice,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))/3600) AS avg_duration_hours
FROM training_jobs
WHERE status IN ('completed', 'failed')
GROUP BY training_mode, status;

-- View: Active learning queue summary
CREATE OR REPLACE VIEW active_learning_summary AS
SELECT 
    query_iteration,
    query_strategy,
    status,
    COUNT(*) AS sample_count,
    AVG(uncertainty_score) AS avg_uncertainty,
    AVG(diversity_score) AS avg_diversity,
    AVG(combined_score) AS avg_combined_score
FROM active_learning_queue
GROUP BY query_iteration, query_strategy, status;

-- ============================================================================
-- Triggers for updated_at Timestamps
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables with updated_at
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_annotations_updated_at
    BEFORE UPDATE ON annotations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Sample Data Insertion (for development/testing)
-- ============================================================================

-- Insert default admin user (password: admin123 - CHANGE IN PRODUCTION)
INSERT INTO users (username, email, hashed_password, role) VALUES
('admin', 'admin@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5ZK5zOPKZ6D3i', 'admin')
ON CONFLICT (username) DO NOTHING;

-- ============================================================================
-- Grants (adjust based on your security requirements)
-- ============================================================================

-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;
