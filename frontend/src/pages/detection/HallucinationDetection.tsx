import React, { useState } from 'react';
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Select,
  Textarea,
  VStack,
  HStack,
  useToast,
  Text,
  Card,
  CardBody,
  CardHeader,
  Heading,
  Badge,
  Alert,
  AlertIcon,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Input,
  Divider,
  CircularProgress,
  CircularProgressLabel,
} from '@chakra-ui/react';
import { useMutation } from '@tanstack/react-query';
import apiClient from '../../api/client';
import { Model, ApiError, getErrorMessage } from '../../api/types';
import { TOAST_DURATION_SUCCESS, TOAST_DURATION_ERROR, TOAST_DURATION_WARNING } from '../../api/constants';

const CONFIDENCE_METHODS = [
  { id: 'geometric', label: 'Geometric Mean', description: 'Sequence probability (default, most robust)' },
  { id: 'average', label: 'Average', description: 'Mean token probability' },
  { id: 'minimum', label: 'Minimum', description: 'Worst-case token confidence (pessimistic)' },
  { id: 'entropy', label: 'Entropy', description: 'Information-theoretic uncertainty' },
  { id: 'variance', label: 'Variance', description: 'Consistency of confidence across tokens' },
];

export default function HallucinationDetection() {
  const toast = useToast();
  const [model, setModel] = useState<Model>({
    name: 'openai:gpt-4',
    description: 'OpenAI GPT-4 for hallucination detection',
    model_name: 'gpt-4',
  });
  const [prompt, setPrompt] = useState('What is the capital of France and when was it founded?');
  const [method, setMethod] = useState('geometric');

  const hallucinationMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await apiClient.post('/hallucination-confidence', data);
      return response.data;
    },
    onSuccess: () => {
      toast({
        title: 'Analysis Complete',
        description: 'Hallucination detection has been completed.',
        status: 'success',
        duration: TOAST_DURATION_SUCCESS,
        isClosable: true,
      });
    },
    onError: (error: ApiError) => {
      toast({
        title: 'Error',
        description: getErrorMessage(error, 'Failed to analyze.'),
        status: 'error',
        duration: TOAST_DURATION_ERROR,
        isClosable: true,
      });
    },
  });

  const handleAnalyze = () => {
    if (!prompt.trim()) {
      toast({ title: 'Please enter a prompt', status: 'warning', duration: TOAST_DURATION_WARNING });
      return;
    }
    hallucinationMutation.mutate({
      model,
      prompt,
      method,
    });
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'green';
      case 'medium': return 'yellow';
      case 'high': return 'orange';
      case 'critical': return 'red';
      default: return 'gray';
    }
  };

  const getConfidenceColor = (score: number) => {
    if (score >= 70) return 'green';
    if (score >= 50) return 'yellow';
    if (score >= 30) return 'orange';
    return 'red';
  };

  const data = hallucinationMutation.data;

  return (
    <Box maxW="7xl" mx="auto" pt={5} px={{ base: 2, sm: 12, md: 17 }}>
      <Text fontSize="2xl" fontWeight="bold" mb={2}>
        Hallucination Detection
      </Text>
      <Text color="gray.600" mb={8}>
        Analyze model confidence using token probabilities to detect potential hallucinations.
      </Text>

      <Card mb={8}>
        <CardHeader>
          <Heading size="md">Configuration</Heading>
        </CardHeader>
        <CardBody>
          <VStack spacing={6} align="stretch">
            <HStack spacing={4}>
              <FormControl flex={2}>
                <FormLabel>Model Name</FormLabel>
                <Select
                  value={model.name}
                  onChange={(e) => setModel({ ...model, name: e.target.value })}
                >
                  <option value="openai:gpt-4">OpenAI GPT-4</option>
                  <option value="openai:gpt-4o">OpenAI GPT-4o</option>
                  <option value="openai:gpt-3.5-turbo">OpenAI GPT-3.5 Turbo</option>
                </Select>
              </FormControl>
              <FormControl flex={2}>
                <FormLabel>Description</FormLabel>
                <Input
                  value={model.description}
                  onChange={(e) => setModel({ ...model, description: e.target.value })}
                  placeholder="Model description"
                />
              </FormControl>
            </HStack>

            <Box>
              <Text fontWeight="semibold" mb={3}>Confidence Calculation Method</Text>
              <SimpleGrid columns={{ base: 1, md: 3, lg: 5 }} spacing={3}>
                {CONFIDENCE_METHODS.map((m) => (
                  <Card
                    key={m.id}
                    cursor="pointer"
                    onClick={() => setMethod(m.id)}
                    borderColor={method === m.id ? 'cyan.500' : 'gray.200'}
                    borderWidth={2}
                    _hover={{ borderColor: 'cyan.300' }}
                  >
                    <CardBody py={3} px={3}>
                      <Text fontSize="sm" fontWeight="semibold">{m.label}</Text>
                      <Text fontSize="xs" color="gray.500">{m.description}</Text>
                    </CardBody>
                  </Card>
                ))}
              </SimpleGrid>
            </Box>

            <FormControl>
              <FormLabel>Prompt</FormLabel>
              <Textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter a factual question to test..."
                rows={4}
              />
            </FormControl>

            <Button
              colorScheme="cyan"
              onClick={handleAnalyze}
              isLoading={hallucinationMutation.isPending}
              loadingText="Analyzing..."
              size="lg"
            >
              Analyze Confidence
            </Button>
          </VStack>
        </CardBody>
      </Card>

      {data && (
        <Card>
          <CardHeader>
            <HStack justify="space-between">
              <Heading size="md">Analysis Results</Heading>
              <Badge fontSize="lg" colorScheme={getRiskColor(data.risk_level)}>
                {data.risk_level?.toUpperCase()} RISK
              </Badge>
            </HStack>
          </CardHeader>
          <CardBody>
            <VStack spacing={6} align="stretch">
              <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6}>
                <Box textAlign="center">
                  <CircularProgress
                    value={data.confidence_score}
                    size="120px"
                    thickness="8px"
                    color={getConfidenceColor(data.confidence_score) + '.400'}
                  >
                    <CircularProgressLabel>
                      <Text fontSize="2xl" fontWeight="bold">
                        {data.confidence_score?.toFixed(1)}%
                      </Text>
                    </CircularProgressLabel>
                  </CircularProgress>
                  <Text mt={2} fontWeight="semibold">Confidence Score</Text>
                </Box>

                <Stat>
                  <StatLabel>Risk Level</StatLabel>
                  <StatNumber>
                    <Badge fontSize="xl" colorScheme={getRiskColor(data.risk_level)}>
                      {data.risk_level?.toUpperCase()}
                    </Badge>
                  </StatNumber>
                  <StatHelpText>Based on confidence thresholds</StatHelpText>
                </Stat>

                <Stat>
                  <StatLabel>Method Used</StatLabel>
                  <StatNumber>
                    <Badge fontSize="lg" colorScheme="cyan">{data.method}</Badge>
                  </StatNumber>
                  <StatHelpText>Confidence calculation method</StatHelpText>
                </Stat>
              </SimpleGrid>

              <Divider />

              <Box>
                <Text fontWeight="semibold" mb={2}>Interpretation</Text>
                <Alert status={data.risk_level === 'low' ? 'success' : data.risk_level === 'medium' ? 'warning' : 'error'}>
                  <AlertIcon />
                  <Text>{data.interpretation}</Text>
                </Alert>
              </Box>

              <Divider />

              <Box>
                <Text fontWeight="semibold" mb={2}>Model Response</Text>
                <Box bg="gray.50" p={4} borderRadius="md">
                  <Text fontSize="sm">{data.response}</Text>
                </Box>
              </Box>

              <Box>
                <Text fontWeight="semibold" mb={2}>Risk Level Guide</Text>
                <SimpleGrid columns={{ base: 2, md: 4 }} spacing={3}>
                  <HStack><Badge colorScheme="green">LOW</Badge><Text fontSize="sm">≥70%</Text></HStack>
                  <HStack><Badge colorScheme="yellow">MEDIUM</Badge><Text fontSize="sm">50-69%</Text></HStack>
                  <HStack><Badge colorScheme="orange">HIGH</Badge><Text fontSize="sm">30-49%</Text></HStack>
                  <HStack><Badge colorScheme="red">CRITICAL</Badge><Text fontSize="sm">&lt;30%</Text></HStack>
                </SimpleGrid>
              </Box>
            </VStack>
          </CardBody>
        </Card>
      )}
    </Box>
  );
}
