package mocks

import (
	"context"
	"database/sql"

	"todo-api/internal/database"
)

// MockResult implements sql.Result for testing
type MockResult struct {
	LastInsertIdValue int64
	RowsAffectedValue int64
}

func (m MockResult) LastInsertId() (int64, error) {
	return m.LastInsertIdValue, nil
}

func (m MockResult) RowsAffected() (int64, error) {
	return m.RowsAffectedValue, nil
}

// MockTx implements the Tx interface for testing transactions
type MockTx struct {
	*MockTodoDB
	CommitFunc   func() error
	RollbackFunc func() error
}

func (m *MockTx) Commit() error {
	if m.CommitFunc != nil {
		return m.CommitFunc()
	}
	return nil
}

func (m *MockTx) Rollback() error {
	if m.RollbackFunc != nil {
		return m.RollbackFunc()
	}
	return nil
}

// MockTodoDB is a mock implementation of the TodoDB interface for testing
type MockTodoDB struct {
	SelectFunc        func(dest interface{}, query string, args ...interface{}) error
	GetFunc           func(dest interface{}, query string, args ...interface{}) error
	ExecFunc          func(query string, args ...interface{}) (sql.Result, error)
	SelectContextFunc func(ctx context.Context, dest interface{}, query string, args ...interface{}) error
	GetContextFunc    func(ctx context.Context, dest interface{}, query string, args ...interface{}) error
	ExecContextFunc   func(ctx context.Context, query string, args ...interface{}) (sql.Result, error)
	BeginxFunc        func() (database.Tx, error)
}

func (m *MockTodoDB) Select(dest interface{}, query string, args ...interface{}) error {
	if m.SelectFunc != nil {
		return m.SelectFunc(dest, query, args...)
	}
	return nil
}

func (m *MockTodoDB) Get(dest interface{}, query string, args ...interface{}) error {
	if m.GetFunc != nil {
		return m.GetFunc(dest, query, args...)
	}
	return nil
}

func (m *MockTodoDB) Exec(query string, args ...interface{}) (sql.Result, error) {
	if m.ExecFunc != nil {
		return m.ExecFunc(query, args...)
	}
	return MockResult{LastInsertIdValue: 1, RowsAffectedValue: 1}, nil
}

func (m *MockTodoDB) SelectContext(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	if m.SelectContextFunc != nil {
		return m.SelectContextFunc(ctx, dest, query, args...)
	}
	return nil
}

func (m *MockTodoDB) GetContext(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	if m.GetContextFunc != nil {
		return m.GetContextFunc(ctx, dest, query, args...)
	}
	return nil
}

func (m *MockTodoDB) ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	if m.ExecContextFunc != nil {
		return m.ExecContextFunc(ctx, query, args...)
	}
	return MockResult{LastInsertIdValue: 1, RowsAffectedValue: 1}, nil
}

func (m *MockTodoDB) Beginx() (database.Tx, error) {
	if m.BeginxFunc != nil {
		return m.BeginxFunc()
	}

	// Return a mock transaction by default
	mockTx := &MockTx{
		MockTodoDB: &MockTodoDB{},
	}
	return mockTx, nil
}
