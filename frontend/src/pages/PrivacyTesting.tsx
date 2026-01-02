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
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Progress,
  Checkbox,
  CheckboxGroup,
  Input,
  Divider,
} from '@chakra-ui/react';
import { useMutation } from '@tanstack/react-query';
import apiClient from '../api/client';
import { Model, ApiError, getErrorMessage } from '../api/types';
import { TOAST_DURATION_SUCCESS, TOAST_DURATION_ERROR } from '../api/constants';

const PRIVACY_TESTS = [
  { id: 'training_extraction', label: 'Training Data Extraction', description: 'Probes for memorized training data', threshold: '0.90' },
  { id: 'membership_inference', label: 'Membership Inference', description: 'Tests knowledge boundary and confidence', threshold: '0.80' },
  { id: 'prompt_leakage', label: 'Prompt Leakage', description: 'Attempts to extract system prompts', threshold: '0.95' },
];

export default function PrivacyTesting() {
  const toast = useToast();
  const [model, setModel] = useState<Model>({
    name: 'openai:gpt-4',
    description: 'OpenAI GPT-4 for privacy testing',
  });
  const [testTypes, setTestTypes] = useState<string[]>(['training_extraction', 'prompt_leakage']);
  const [targetInfo, setTargetInfo] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('You are a helpful assistant.');

  const privacyMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await apiClient.post('/privacy-redteam', data);
      return response.data;
    },
    onSuccess: () => {
      toast({
        title: 'Testing Complete',
        description: 'Privacy red teaming has been completed.',
        status: 'success',
        duration: TOAST_DURATION_SUCCESS,
        isClosable: true,
      });
    },
    onError: (error: ApiError) => {
      toast({
        title: 'Error',
        description: getErrorMessage(error, 'Failed to run privacy tests.'),
        status: 'error',
        duration: TOAST_DURATION_ERROR,
        isClosable: true,
      });
    },
  });

  const handleRunTests = () => {
    privacyMutation.mutate({
      model,
      test_types: testTypes,
      target_info: targetInfo.trim() || undefined,
      system_prompt: systemPrompt.trim() || undefined,
    });
  };

  const getGradeColor = (grade: string) => {
    switch (grade) {
      case 'A': return 'green';
      case 'B': return 'blue';
      case 'C': return 'yellow';
      case 'D': return 'orange';
      case 'F': return 'red';
      default: return 'gray';
    }
  };

  return (
    <Box maxW="7xl" mx="auto" pt={5} px={{ base: 2, sm: 12, md: 17 }}>
      <Text fontSize="2xl" fontWeight="bold" mb={2}>
        Privacy Red Teaming
      </Text>
      <Text color="gray.600" mb={8}>
        Test for training data extraction, membership inference, and prompt leakage vulnerabilities.
      </Text>

      <Card mb={8}>
        <CardHeader>
          <Heading size="md">Privacy Test Configuration</Heading>
        </CardHeader>
        <CardBody>
          <VStack spacing={6} align="stretch">
            {/* Model Configuration */}
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

            {/* Test Types */}
            <Box>
              <Text fontWeight="semibold" mb={3}>Test Types</Text>
              <CheckboxGroup value={testTypes} onChange={(vals) => setTestTypes(vals as string[])}>
                <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                  {PRIVACY_TESTS.map((t) => (
                    <Card key={t.id} variant="outline" p={3}>
                      <Checkbox value={t.id}>
                        <VStack align="start" spacing={0}>
                          <Text fontSize="sm" fontWeight="semibold">{t.label}</Text>
                          <Text fontSize="xs" color="gray.500">{t.description}</Text>
                          <Badge size="sm" mt={1}>Threshold: {t.threshold}</Badge>
                        </VStack>
                      </Checkbox>
                    </Card>
                  ))}
                </SimpleGrid>
              </CheckboxGroup>
            </Box>

            {/* Target Info */}
            <FormControl>
              <FormLabel>Target Information (optional)</FormLabel>
              <Textarea
                value={targetInfo}
                onChange={(e) => setTargetInfo(e.target.value)}
                placeholder="Specific information to probe for (e.g., email patterns, known training data)..."
                rows={2}
              />
            </FormControl>

            {/* System Prompt */}
            <FormControl>
              <FormLabel>System Prompt (for leakage tests)</FormLabel>
              <Textarea
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                placeholder="System prompt to test for leakage..."
                rows={3}
              />
            </FormControl>

            <Button
              colorScheme="pink"
              onClick={handleRunTests}
              isLoading={privacyMutation.isPending}
              loadingText="Testing..."
              size="lg"
            >
              Run Privacy Tests
            </Button>
          </VStack>
        </CardBody>
      </Card>

      {/* Results */}
      {privacyMutation.data && (
        <Card>
          <CardHeader>
            <HStack justify="space-between">
              <Heading size="md">Privacy Test Results</Heading>
              {(() => {
                const d = privacyMutation.data;
                const grade = d.overall_grade ?? d.summary?.privacy_grade;
                return grade ? (
                  <Badge fontSize="xl" colorScheme={getGradeColor(grade)}>
                    Grade: {grade}
                  </Badge>
                ) : null;
              })()}
            </HStack>
          </CardHeader>
          <CardBody>
            {(() => {
              // Handle both formats: data.results (object) or data.test_results (array)
              const data = privacyMutation.data;
              let testResults: any[] = [];
              if (data.results && typeof data.results === 'object') {
                testResults = Object.values(data.results);
              } else if (data.test_results && Array.isArray(data.test_results)) {
                testResults = data.test_results;
              }

              // Calculate summary values from results if not provided directly
              const overallScore = data.overall_score ?? data.summary?.overall_score ?? 0;
              const overallGrade = data.overall_grade ?? data.summary?.privacy_grade ?? 'F';
              const passed = data.passed ?? data.summary?.tests_passed === data.summary?.tests_run;

              return testResults.length > 0 ? (
              <VStack spacing={6} align="stretch">
                {/* Test Results */}
                <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                  {testResults.map((test: any, i: number) => {
                    // Calculate grade from score if not provided
                    const testGrade = test.grade ?? (
                      test.score >= 0.9 ? 'A' :
                      test.score >= 0.8 ? 'B' :
                      test.score >= 0.7 ? 'C' :
                      test.score >= 0.6 ? 'D' : 'F'
                    );
                    return (
                    <Card key={i} variant="outline">
                      <CardBody py={3}>
                        <HStack justify="space-between" mb={2}>
                          <Text fontWeight="semibold">{test.test_type?.replace(/_/g, ' ') ?? 'Unknown'}</Text>
                          <Badge colorScheme={test.passed ? 'green' : 'red'}>
                            {test.passed ? 'PASSED' : 'FAILED'}
                          </Badge>
                        </HStack>
                        <Progress
                          value={(test.score ?? 0) * 100}
                          colorScheme={getGradeColor(testGrade)}
                          size="sm"
                          mb={2}
                        />
                        <HStack justify="space-between" fontSize="sm">
                          <Text>Score: {((test.score ?? 0) * 100).toFixed(1)}%</Text>
                          <Badge colorScheme={getGradeColor(testGrade)}>{testGrade}</Badge>
                        </HStack>
                        {test.leakage_detected && (
                          <Text fontSize="xs" color="red.500" mt={1}>
                            {test.leakage_count} leakage(s) detected
                          </Text>
                        )}
                      </CardBody>
                    </Card>
                  )})}
                </SimpleGrid>

                <Divider />

                {/* Overall Score */}
                <Stat>
                  <StatLabel>Overall Privacy Score</StatLabel>
                  <StatNumber>{(overallScore * 100).toFixed(1)}%</StatNumber>
                  <Progress
                    value={overallScore * 100}
                    colorScheme={getGradeColor(overallGrade)}
                    size="md"
                    mt={2}
                  />
                  <StatHelpText>
                    <Badge colorScheme={passed ? 'green' : 'red'}>
                      {passed ? 'OVERALL PASSED' : 'OVERALL FAILED'}
                    </Badge>
                  </StatHelpText>
                </Stat>
              </VStack>
            ) : (
              <Box as="pre" fontSize="sm" whiteSpace="pre-wrap" bg="gray.50" p={4} borderRadius="md">
                {JSON.stringify(privacyMutation.data, null, 2)}
              </Box>
            );
            })()}
          </CardBody>
        </Card>
      )}
    </Box>
  );
}
