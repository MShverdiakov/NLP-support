package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/joho/godotenv"
)

// hnowledge db
var knowledgeBase = map[string]string{
	"order_status":    "Your order is being processed. Please check your email for updates.",
	"refund_policy":   "Our refund policy is available on our website under 'Refunds'.",
	"product_info":    "We offer a wide range of products to meet your needs. Visit our product page for details.",
	"shipping_info":   "We ship worldwide. Delivery takes 3-5 business days.",
	"payment_methods": "We accept Visa, MasterCard, and PayPal.",
}

// QueryProcessor handles NLP query processing with rate limiting and concurrency control
type QueryProcessor struct {
	apiKey      string
	rateLimiter chan struct{} // Token bucket for rate limiting
	client      *http.Client
	maxRetries  int
}

type OpenRouterRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenRouterResponse struct {
	Choices []Choice `json:"choices"`
}

type Choice struct {
	Message Message `json:"message"`
}

type ErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

// Result represents the processed query result
type Result struct {
	Query    string
	Response string
	Error    error
}

// NewQueryProcessor creates a new QueryProcessor with rate limiting
func NewQueryProcessor(apiKey string, maxConcurrent int) *QueryProcessor {
	return &QueryProcessor{
		apiKey:      apiKey,
		rateLimiter: make(chan struct{}, maxConcurrent),
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		maxRetries: 3,
	}
}

// ProcessQueries processes multiple queries concurrently with rate limiting
func (p *QueryProcessor) ProcessQueries(ctx context.Context, queries []string) []Result {
	results := make([]Result, len(queries))
	var wg sync.WaitGroup

	// Create a worker pool
	for i, query := range queries {
		wg.Add(1)
		go func(i int, query string) {
			defer wg.Done()

			// Implement rate limiting
			select {
			case p.rateLimiter <- struct{}{}: // Get token
				defer func() { <-p.rateLimiter }() // Release token
			case <-ctx.Done():
				results[i] = Result{Query: query, Error: ctx.Err()}
				return
			}

			// retries
			response, err := p.processQueryWithRetry(ctx, query)
			results[i] = Result{
				Query:    query,
				Response: response,
				Error:    err,
			}
		}(i, query)
	}

	wg.Wait()
	return results
}

func (p *QueryProcessor) processQueryWithRetry(ctx context.Context, query string) (string, error) {
	var lastErr error
	for attempt := 0; attempt < p.maxRetries; attempt++ {
		if response, err := p.processQuery(ctx, query); err != nil {
			lastErr = err                                                 // Store the last error encountered
			time.Sleep(time.Duration(attempt+1) * 500 * time.Millisecond) // Wait before retrying
		} else {
			return response, nil
		}
	}
	return "", fmt.Errorf("max retries exceeded: %v", lastErr)
}

// processQuery sends a request to check intent and returns response
func (p *QueryProcessor) processQuery(ctx context.Context, query string) (string, error) {
	// Check intent using NLP
	intent, err := p.checkIntent(ctx, query)
	if err != nil {
		return "", fmt.Errorf("intent check failed: %v", err)
	}

	// Look for answer in knowledge base
	if answer, exists := knowledgeBase[intent]; exists {
		return answer, nil
	}

	// Escalate to human if no answer found
	return "I'll connect you with a human agent for better assistance. Please wait.", nil
}

// checkIntent uses NLP to determine query intent
func (p *QueryProcessor) checkIntent(ctx context.Context, query string) (string, error) {
	reqBody := OpenRouterRequest{
		Model: "deepseek/deepseek-r1:free",
		Messages: []Message{
			{
				Role:    "system",
				Content: "Analyze the query and return ONLY ONE of these intents: order_status, refund_policy, product_info, shipping_info, payment_methods, or 'escalate' if none match.",
			},
			{
				Role:    "user",
				Content: query,
			},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %v", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://openrouter.ai/api/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", p.apiKey))
	req.Header.Set("HTTP-Referer", "http://localhost:8080")
	req.Header.Set("X-Title", "NLP Query Service")

	resp, err := p.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errorResp ErrorResponse
		if err := json.Unmarshal(body, &errorResp); err != nil {
			return "", fmt.Errorf("API returned non-200 status code: %d, body: %s", resp.StatusCode, string(body))
		}
		return "", fmt.Errorf("API error: %s (type: %s, code: %s)",
			errorResp.Error.Message,
			errorResp.Error.Type,
			errorResp.Error.Code)
	}

	var openRouterResp OpenRouterResponse
	if err := json.Unmarshal(body, &openRouterResp); err != nil {
		return "", fmt.Errorf("failed to decode response: %v", err)
	}

	if len(openRouterResp.Choices) == 0 {
		return "escalate", nil
	}

	intent := strings.ToLower(strings.TrimSpace(openRouterResp.Choices[0].Message.Content))
	if _, exists := knowledgeBase[intent]; !exists {
		return "escalate", nil
	}

	return intent, nil
}

func main() {
	// loading env variables
	if err := godotenv.Load(); err != nil {
		log.Printf("Warning: .env file not found: %v", err)
	}

	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENROUTER_API_KEY environment variable is not set")
	}

	// Example queries to test
	queries := []string{
		"How do I reset my password?",
		"I need help with my order",
		"What are your business hours?",
		// Add more queries here
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	// limit of concurrent queries
	processor := NewQueryProcessor(apiKey, 5)

	// Processing concurrently
	results := processor.ProcessQueries(ctx, queries)

	// results
	for _, result := range results {
		fmt.Printf("\nQuery: %s\n", result.Query)
		if result.Error != nil {
			fmt.Printf("Error: %v\n", result.Error)
		} else {
			fmt.Printf("Response: %s\n", result.Response)
		}
	}
}

/* Масштабирование и NLP:
   - Добавить кэширование частых запросов
   - Использовать очереди для распределения нагрузки
   - Интегрировать с NLP движком
   - Мониторинг производительности
   - Балансировка между сервисами
*/
