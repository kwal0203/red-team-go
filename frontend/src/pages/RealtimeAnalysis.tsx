import React, { useState } from 'react';
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Input,
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
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Badge,
  Progress,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  SimpleGrid,
  Alert,
  AlertIcon,
  Divider,
} from '@chakra-ui/react';
import { useMutation } from '@tanstack/react-query';
import apiClient from '../api/client';
import { DetectionRealtimeRequest, ToxicityRealtimeResult, BiasRealtimeResult, ApiError, getErrorMessage } from '../api/types';
import { TOAST_DURATION_SUCCESS, TOAST_DURATION_ERROR, TOAST_DURATION_WARNING } from '../api/constants';
import { useModelSelector, MODEL_OPTIONS } from '../hooks';

export default function RealtimeAnalysis() {
  const toast = useToast();
  const { model, setModel, handleModelChange } = useModelSelector({
    defaultDescription: 'OpenAI GPT-4o-mini for realtime analysis',
  });
  const [prompt, setPrompt] = useState('');

  const toxicityMutation = useMutation({
    mutationFn: async (data: DetectionRealtimeRequest) => {
      const response = await apiClient.post('/toxicity-detection-realtime', data);
      return response.data as ToxicityRealtimeResult;
    },
    onSuccess: () => {
      toast({
        title: 'Analysis Complete',
        description: 'Toxicity analysis has been completed successfully.',
        status: 'success',
        duration: TOAST_DURATION_SUCCESS,
        isClosable: true,
      });
    },
    onError: (error: ApiError) => {
      toast({
        title: 'Error',
        description: getErrorMessage(error, 'Failed to complete toxicity analysis.'),
        status: 'error',
        duration: TOAST_DURATION_ERROR,
        isClosable: true,
      });
    },
  });

  const biasMutation = useMutation({
    mutationFn: async (data: DetectionRealtimeRequest) => {
      const response = await apiClient.post('/bias-detection-realtime', data);
      return response.data as BiasRealtimeResult;
    },
    onSuccess: () => {
      toast({
        title: 'Analysis Complete',
        description: 'Bias analysis has been completed successfully.',
        status: 'success',
        duration: TOAST_DURATION_SUCCESS,
        isClosable: true,
      });
    },
    onError: (error: ApiError) => {
      toast({
        title: 'Error',
        description: getErrorMessage(error, 'Failed to complete bias analysis.'),
        status: 'error',
        duration: TOAST_DURATION_ERROR,
        isClosable: true,
      });
    },
  });

  const handleToxicityAnalysis = () => {
    if (prompt.trim()) {
      toxicityMutation.mutate({ model, prompt });
    } else {
      toast({
        title: 'Validation Error',
        description: 'Please enter a prompt.',
        status: 'warning',
        duration: TOAST_DURATION_WARNING,
      });
    }
  };

  const handleBiasAnalysis = () => {
    if (prompt.trim()) {
      biasMutation.mutate({ model, prompt });
    } else {
      toast({
        title: 'Validation Error',
        description: 'Please enter a prompt.',
        status: 'warning',
        duration: TOAST_DURATION_WARNING,
      });
    }
  };

  const handleBothAnalysis = () => {
    if (prompt.trim()) {
      toxicityMutation.mutate({ model, prompt });
      biasMutation.mutate({ model, prompt });
    } else {
      toast({
        title: 'Validation Error',
        description: 'Please enter a prompt.',
        status: 'warning',
        duration: TOAST_DURATION_WARNING,
      });
    }
  };

  const getToxicityColor = (score: number) => {
    if (score < 0.3) return 'green';
    if (score < 0.6) return 'yellow';
    return 'red';
  };

  return (
    <Box maxW="7xl" mx="auto" pt={5} px={{ base: 2, sm: 12, md: 17 }}>
      <Text fontSize="2xl" fontWeight="bold" mb={2}>
        Real-time Analysis
      </Text>
      <Text color="gray.600" mb={8}>
        Analyze a single prompt for toxicity and bias in real-time.
      </Text>

      <Card mb={8}>
        <CardHeader>
          <Heading size="md">Configuration</Heading>
        </CardHeader>
        <CardBody>
          <VStack spacing={6} align="stretch">
            {/* Model Configuration */}
            <Box>
              <Text fontWeight="semibold" mb={3}>Target Model</Text>
              <HStack spacing={4}>
                <FormControl flex={2}>
                  <FormLabel>Model Name</FormLabel>
                  <Select
                    value={model.name}
                    onChange={(e) => handleModelChange(e.target.value)}
                  >
                    {MODEL_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </Select>
                </FormControl>
                <FormControl flex={3}>
                  <FormLabel>Description</FormLabel>
                  <Input
                    value={model.description}
                    onChange={(e) => setModel({ ...model, description: e.target.value })}
                    placeholder="Model description"
                  />
                </FormControl>
              </HStack>
            </Box>

            {/* Prompt Input */}
            <FormControl>
              <FormLabel>Prompt</FormLabel>
              <Textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter your prompt here to test for toxicity and bias..."
                size="lg"
                rows={4}
              />
            </FormControl>

            {/* Action Buttons */}
            <HStack spacing={4}>
              <Button
                colorScheme="blue"
                onClick={handleToxicityAnalysis}
                isLoading={toxicityMutation.isPending}
                loadingText="Analyzing..."
                flex={1}
              >
                Toxicity Only
              </Button>
              <Button
                colorScheme="purple"
                onClick={handleBiasAnalysis}
                isLoading={biasMutation.isPending}
                loadingText="Analyzing..."
                flex={1}
              >
                Bias Only
              </Button>
              <Button
                colorScheme="teal"
                onClick={handleBothAnalysis}
                isLoading={toxicityMutation.isPending || biasMutation.isPending}
                loadingText="Analyzing..."
                flex={1}
              >
                Run Both
              </Button>
            </HStack>
          </VStack>
        </CardBody>
      </Card>

      {/* Results */}
      <Tabs colorScheme="blue">
        <TabList>
          <Tab>
            Toxicity Results
            {toxicityMutation.data && (
              <Badge ml={2} colorScheme={toxicityMutation.data.toxicity?.is_toxic ? 'red' : 'green'}>
                {toxicityMutation.data.toxicity?.is_toxic ? 'Toxic' : 'Safe'}
              </Badge>
            )}
          </Tab>
          <Tab>
            Bias Results
            {biasMutation.data && (
              <Badge ml={2} colorScheme={biasMutation.data.bias?.bias_detected ? 'red' : 'green'}>
                {biasMutation.data.bias?.bias_detected ? 'Biased' : 'Neutral'}
              </Badge>
            )}
          </Tab>
        </TabList>

        <TabPanels>
          {/* Toxicity Tab */}
          <TabPanel px={0}>
            {toxicityMutation.data ? (
              <Card>
                <CardBody>
                  <VStack spacing={6} align="stretch">
                    {/* Scores */}
                    <SimpleGrid columns={{ base: 1, md: 4 }} spacing={4}>
                      <Stat>
                        <StatLabel>OpenAI Score</StatLabel>
                        <StatNumber>{((toxicityMutation.data.toxicity?.openai_score || 0) * 100).toFixed(1)}%</StatNumber>
                        <Progress
                          value={(toxicityMutation.data.toxicity?.openai_score || 0) * 100}
                          colorScheme={getToxicityColor(toxicityMutation.data.toxicity?.openai_score || 0)}
                          size="sm"
                          mt={2}
                        />
                      </Stat>
                      <Stat>
                        <StatLabel>Paradetox Score</StatLabel>
                        <StatNumber>{((toxicityMutation.data.toxicity?.paradetox_score || 0) * 100).toFixed(1)}%</StatNumber>
                        <Progress
                          value={(toxicityMutation.data.toxicity?.paradetox_score || 0) * 100}
                          colorScheme={getToxicityColor(toxicityMutation.data.toxicity?.paradetox_score || 0)}
                          size="sm"
                          mt={2}
                        />
                      </Stat>
                      <Stat>
                        <StatLabel>Average Score</StatLabel>
                        <StatNumber>{((toxicityMutation.data.toxicity?.average_score || 0) * 100).toFixed(1)}%</StatNumber>
                        <Progress
                          value={(toxicityMutation.data.toxicity?.average_score || 0) * 100}
                          colorScheme={getToxicityColor(toxicityMutation.data.toxicity?.average_score || 0)}
                          size="sm"
                          mt={2}
                        />
                      </Stat>
                      <Stat>
                        <StatLabel>Status</StatLabel>
                        <StatNumber>
                          <Badge
                            fontSize="lg"
                            colorScheme={toxicityMutation.data.toxicity?.is_toxic ? 'red' : 'green'}
                          >
                            {toxicityMutation.data.toxicity?.is_toxic ? 'TOXIC' : 'SAFE'}
                          </Badge>
                        </StatNumber>
                        <StatHelpText>Threshold: 50%</StatHelpText>
                      </Stat>
                    </SimpleGrid>

                    <Divider />

                    {/* Model Response */}
                    <Box>
                      <Text fontWeight="semibold" mb={2}>Model Response:</Text>
                      <Box bg="gray.50" p={4} borderRadius="md">
                        <Text fontSize="sm">{toxicityMutation.data.model_response}</Text>
                      </Box>
                    </Box>
                  </VStack>
                </CardBody>
              </Card>
            ) : (
              <Alert status="info">
                <AlertIcon />
                Run a toxicity analysis to see results here.
              </Alert>
            )}
          </TabPanel>

          {/* Bias Tab */}
          <TabPanel px={0}>
            {biasMutation.data ? (
              <Card>
                <CardBody>
                  <VStack spacing={6} align="stretch">
                    {/* Status */}
                    <HStack>
                      <Text fontWeight="semibold">Bias Status:</Text>
                      <Badge
                        fontSize="lg"
                        colorScheme={biasMutation.data.bias?.bias_detected ? 'red' : 'green'}
                      >
                        {biasMutation.data.bias?.bias_detected ? 'BIAS DETECTED' : 'NO BIAS DETECTED'}
                      </Badge>
                    </HStack>

                    <Divider />

                    {/* Model Response */}
                    <Box>
                      <Text fontWeight="semibold" mb={2}>Model Response:</Text>
                      <Box bg="gray.50" p={4} borderRadius="md">
                        <Text fontSize="sm">{biasMutation.data.model_response}</Text>
                      </Box>
                    </Box>

                    {/* Explanation */}
                    <Box>
                      <Text fontWeight="semibold" mb={2}>Analysis:</Text>
                      <Box bg={biasMutation.data.bias?.bias_detected ? 'red.50' : 'green.50'} p={4} borderRadius="md">
                        <Text fontSize="sm">{biasMutation.data.bias?.explanation}</Text>
                      </Box>
                    </Box>
                  </VStack>
                </CardBody>
              </Card>
            ) : (
              <Alert status="info">
                <AlertIcon />
                Run a bias analysis to see results here.
              </Alert>
            )}
          </TabPanel>
        </TabPanels>
      </Tabs>

      {/* Error Display */}
      {(toxicityMutation.error || biasMutation.error) && (
        <Alert status="error" mt={4}>
          <AlertIcon />
          {getErrorMessage(toxicityMutation.error as ApiError, '') ||
           getErrorMessage(biasMutation.error as ApiError, '') ||
           'An error occurred during analysis.'}
        </Alert>
      )}
    </Box>
  );
}
