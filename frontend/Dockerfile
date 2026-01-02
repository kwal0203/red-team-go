# Build stage
FROM node:18.19.0-alpine3.19 AS builder

WORKDIR /app

# Install dependencies required for building node modules
RUN apk add --no-cache python3 make g++

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install --legacy-peer-deps

# Copy the rest of the application
COPY . .

# Allow the API base URL to be baked into the frontend build.
ARG REACT_APP_API_URL
ENV REACT_APP_API_URL=${REACT_APP_API_URL}

# Build the application
RUN npm run build

# Runtime stage
FROM node:18.19.0-alpine3.19

WORKDIR /app

# Install serve to run the built application
RUN npm install -g serve

# Copy the built application from builder stage
COPY --from=builder /app/build ./build

# Expose port
EXPOSE 3000

# Start the application with single-page application support
CMD ["serve", "-s", "build", "-l", "3000", "--single"]
