-- Database initialization script for Todo API
-- This script runs automatically when the PostgreSQL container starts

-- Create todos table with proper indexing and constraints
CREATE TABLE todos (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL CHECK (length(title) > 0),
    description TEXT,
    completed BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create index for common query patterns
CREATE INDEX idx_todos_completed ON todos(completed);
CREATE INDEX idx_todos_created_at ON todos(created_at);
CREATE INDEX idx_todos_title ON todos USING gin(to_tsvector('english', title));

-- Create a function to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at on row changes
CREATE TRIGGER update_todos_updated_at
    BEFORE UPDATE ON todos
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data for testing
INSERT INTO todos (title, description, completed) VALUES
    ('Learn Go basics', 'Study Go syntax, types, and basic programming concepts', false),
    ('Set up PostgreSQL', 'Configure PostgreSQL database with Docker for persistent storage', true),
    ('Implement REST API', 'Build RESTful endpoints for todo CRUD operations', false),
    ('Add authentication', 'Implement JWT-based authentication for API security', false),
    ('Deploy to production', 'Set up CI/CD pipeline and deploy to cloud provider', false);

-- Create a schema_versions table for migration tracking
CREATE TABLE schema_versions (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Record the initial schema version
INSERT INTO schema_versions (version, description) VALUES
    (1, 'Initial schema with todos table and basic indexes');