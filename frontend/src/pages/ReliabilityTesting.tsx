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
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Badge,
  Alert,
  AlertIcon,
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
import { Model } from '../api/types';

const CONSISTENCY_TESTS = [
  { id: 'sycophancy', label: 'Sycophancy', description: 'Tests if model changes opinions when challenged' },
  { id: 'stability', label: 'Stability', description: 'Tests consistency under prompt paraphrasing' },
  { id: 'self_consistency', label: 'Self-Consistency', description: 'Tests consistency across multiple generations' },
  { id: 'instruction_following', label: 'Instruction Following', description: 'Tests adherence to formatting constraints' },
];

const MISINFORMATION_TESTS = [
  { id: 'knowledge_cutoff', label: 'Knowledge Cutoff', description: 'Tests if model acknowledges training date limits' },
  { id: 'temporal_reasoning', label: 'Temporal Reasoning', description: 'Tests date/time arithmetic and ordering' },
  { id: 'confidence_calibration', label: 'Confidence Calibration', description: 'Tests if expressed confidence matches accuracy' },
  { id: 'citation_verification', label: 'Citation Verification', description: 'Tests self-consistency of cited claims' },
];

const REFUSAL_TESTS = [
  { id: 'paraphrase', label: 'Paraphrase', description: 'Consistency across paraphrased harmful requests' },
  { id: 'pressure', label: 'Pressure', description: 'Resistance to user manipulation tactics' },
  { id: 'multi_turn', label: 'Multi-turn', description: 'Resistance to gradual trust-building attacks' },
  { id: 'context_switching', label: 'Context Switching', description: 'Detection of harmful content in benign context' },
];

const HARM_CATEGORIES = ['weapons', 'illegal_activities', 'self_harm', 'violence', 'cbrn', 'dual_use'];

export default function ReliabilityTesting() {
  const toast = useToast();
  const [model, setModel] = useState<Model>({
    name: 'openai:gpt-4',
    description: 'OpenAI GPT-4 for reliability testing',
  });
  const [prompt, setPrompt] = useState('');

  // Consistency state
  const [consistencyTests, setConsistencyTests] = useState<string[]>(['sycophancy', 'stability']);

  // Misinformation state
  const [misinfoTests, setMisinfoTests] = useState<string[]>(['knowledge_cutoff', 'temporal_reasoning']);

  // Refusal state
  const [refusalTests, setRefusalTests] = useState<string[]>(['paraphrase', 'pressure']);
  const [harmCategory, setHarmCategory] = useState('weapons');

  const consistencyMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await apiClient.post('/consistency-reliability', data);
      return response.data;
    },
    onSuccess: () => {
      toast({ title: 'Testing Complete', status: 'success', duration: 3000, isClosable: true });
    },
    onError: (error: any) => {
      toast({ title: 'Error', description: error.response?.data?.detail || 'Failed', status: 'error', duration: 5000 });
    },
  });

  const misinfoMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await apiClient.post('/misinformation-factuality', data);
      return response.data;
    },
    onSuccess: () => {
      toast({ title: 'Testing Complete', status: 'success', duration: 3000, isClosable: true });
    },
    onError: (error: any) => {
      toast({ title: 'Error', description: error.response?.data?.detail || 'Failed', status: 'error', duration: 5000 });
    },
  });

  const refusalMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await apiClient.post('/refusal-consistency', data);
      return response.data;
    },
    onSuccess: () => {
      toast({ title: 'Testing Complete', status: 'success', duration: 3000, isClosable: true });
    },
    onError: (error: any) => {
      toast({ title: 'Error', description: error.response?.data?.detail || 'Failed', status: 'error', duration: 5000 });
    },
  });

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

  const renderTestResults = (data: any) => {
    if (!data) return null;

    return (
      <Card>
        <CardHeader>
          <HStack justify="space-between">
            <Heading size="md">Test Results</Heading>
            {data.overall_grade && (
              <Badge fontSize="xl" colorScheme={getGradeColor(data.overall_grade)}>
                Grade: {data.overall_grade}
              </Badge>
            )}
          </HStack>
        </CardHeader>
        <CardBody>
          {data.test_results ? (
            <VStack spacing={4} align="stretch">
              <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                {data.test_results.map((test: any, i: number) => (
                  <Card key={i} variant="outline">
                    <CardBody py={3}>
                      <HStack justify="space-between" mb={2}>
                        <Text fontWeight="semibold">{test.test_type}</Text>
                        <Badge colorScheme={test.passed ? 'green' : 'red'}>
                          {test.passed ? 'PASSED' : 'FAILED'}
                        </Badge>
                      </HStack>
                      <Progress
                        value={test.score * 100}
                        colorScheme={getGradeColor(test.grade)}
                        size="sm"
                        mb={2}
                      />
                      <HStack justify="space-between" fontSize="sm">
                        <Text>Score: {(test.score * 100).toFixed(1)}%</Text>
                        <Badge colorScheme={getGradeColor(test.grade)}>{test.grade}</Badge>
                      </HStack>
                    </CardBody>
                  </Card>
                ))}
              </SimpleGrid>

              <Divider />

              <Stat>
                <StatLabel>Overall Score</StatLabel>
                <StatNumber>{(data.overall_score * 100).toFixed(1)}%</StatNumber>
                <Progress
                  value={data.overall_score * 100}
                  colorScheme={getGradeColor(data.overall_grade)}
                  size="md"
                  mt={2}
                />
                <StatHelpText>
                  <Badge colorScheme={data.passed ? 'green' : 'red'}>
                    {data.passed ? 'OVERALL PASSED' : 'OVERALL FAILED'}
                  </Badge>
                </StatHelpText>
              </Stat>
            </VStack>
          ) : (
            <Box as="pre" fontSize="sm" whiteSpace="pre-wrap" bg="gray.50" p={4} borderRadius="md">
              {JSON.stringify(data, null, 2)}
            </Box>
          )}
        </CardBody>
      </Card>
    );
  };

  return (
    <Box maxW="7xl" mx="auto" pt={5} px={{ base: 2, sm: 12, md: 17 }}>
      <Text fontSize="2xl" fontWeight="bold" mb={2}>
        Reliability Testing
      </Text>
      <Text color="gray.600" mb={8}>
        Test model consistency, factuality, and refusal behavior.
      </Text>

      <Tabs colorScheme="green">
        <TabList>
          <Tab>Consistency</Tab>
          <Tab>Misinformation</Tab>
          <Tab>Refusal</Tab>
        </TabList>

        <TabPanels>
          {/* Consistency Tab */}
          <TabPanel px={0}>
            <Card mb={8}>
              <CardHeader>
                <Heading size="md">Consistency & Reliability</Heading>
              </CardHeader>
              <CardBody>
                <VStack spacing={6} align="stretch">
                  <HStack spacing={4}>
                    <FormControl flex={2}>
                      <FormLabel>Model</FormLabel>
                      <Select value={model.name} onChange={(e) => setModel({ ...model, name: e.target.value })}>
                        <option value="openai:gpt-4">OpenAI GPT-4</option>
                        <option value="openai:gpt-4o">OpenAI GPT-4o</option>
                        <option value="openai:gpt-3.5-turbo">OpenAI GPT-3.5 Turbo</option>
                      </Select>
                    </FormControl>
                  </HStack>

                  <Box>
                    <Text fontWeight="semibold" mb={3}>Test Types</Text>
                    <CheckboxGroup value={consistencyTests} onChange={(v) => setConsistencyTests(v as string[])}>
                      <SimpleGrid columns={{ base: 1, md: 2 }} spacing={2}>
                        {CONSISTENCY_TESTS.map((t) => (
                          <Checkbox key={t.id} value={t.id}>
                            <Text fontSize="sm">{t.label}</Text>
                          </Checkbox>
                        ))}
                      </SimpleGrid>
                    </CheckboxGroup>
                  </Box>

                  <FormControl>
                    <FormLabel>Test Prompt</FormLabel>
                    <Textarea
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder="Enter a prompt to test consistency..."
                      rows={3}
                    />
                  </FormControl>

                  <Button
                    colorScheme="green"
                    onClick={() => consistencyMutation.mutate({ model, prompt, test_types: consistencyTests })}
                    isLoading={consistencyMutation.isPending}
                    size="lg"
                  >
                    Run Consistency Tests
                  </Button>
                </VStack>
              </CardBody>
            </Card>
            {renderTestResults(consistencyMutation.data)}
          </TabPanel>

          {/* Misinformation Tab */}
          <TabPanel px={0}>
            <Card mb={8}>
              <CardHeader>
                <Heading size="md">Misinformation & Factuality</Heading>
              </CardHeader>
              <CardBody>
                <VStack spacing={6} align="stretch">
                  <HStack spacing={4}>
                    <FormControl flex={2}>
                      <FormLabel>Model</FormLabel>
                      <Select value={model.name} onChange={(e) => setModel({ ...model, name: e.target.value })}>
                        <option value="openai:gpt-4">OpenAI GPT-4</option>
                        <option value="openai:gpt-4o">OpenAI GPT-4o</option>
                        <option value="openai:gpt-3.5-turbo">OpenAI GPT-3.5 Turbo</option>
                      </Select>
                    </FormControl>
                  </HStack>

                  <Box>
                    <Text fontWeight="semibold" mb={3}>Test Types</Text>
                    <CheckboxGroup value={misinfoTests} onChange={(v) => setMisinfoTests(v as string[])}>
                      <SimpleGrid columns={{ base: 1, md: 2 }} spacing={2}>
                        {MISINFORMATION_TESTS.map((t) => (
                          <Checkbox key={t.id} value={t.id}>
                            <Text fontSize="sm">{t.label}</Text>
                          </Checkbox>
                        ))}
                      </SimpleGrid>
                    </CheckboxGroup>
                  </Box>

                  <FormControl>
                    <FormLabel>Test Prompt</FormLabel>
                    <Textarea
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder="Enter a prompt to test factuality..."
                      rows={3}
                    />
                  </FormControl>

                  <Button
                    colorScheme="green"
                    onClick={() => misinfoMutation.mutate({ model, prompt, test_types: misinfoTests })}
                    isLoading={misinfoMutation.isPending}
                    size="lg"
                  >
                    Run Factuality Tests
                  </Button>
                </VStack>
              </CardBody>
            </Card>
            {renderTestResults(misinfoMutation.data)}
          </TabPanel>

          {/* Refusal Tab */}
          <TabPanel px={0}>
            <Card mb={8}>
              <CardHeader>
                <Heading size="md">Refusal Consistency</Heading>
              </CardHeader>
              <CardBody>
                <VStack spacing={6} align="stretch">
                  <HStack spacing={4}>
                    <FormControl flex={2}>
                      <FormLabel>Model</FormLabel>
                      <Select value={model.name} onChange={(e) => setModel({ ...model, name: e.target.value })}>
                        <option value="openai:gpt-4">OpenAI GPT-4</option>
                        <option value="openai:gpt-4o">OpenAI GPT-4o</option>
                        <option value="openai:gpt-3.5-turbo">OpenAI GPT-3.5 Turbo</option>
                      </Select>
                    </FormControl>
                    <FormControl flex={1}>
                      <FormLabel>Harm Category</FormLabel>
                      <Select value={harmCategory} onChange={(e) => setHarmCategory(e.target.value)}>
                        {HARM_CATEGORIES.map((c) => (
                          <option key={c} value={c}>{c.replace('_', ' ')}</option>
                        ))}
                      </Select>
                    </FormControl>
                  </HStack>

                  <Box>
                    <Text fontWeight="semibold" mb={3}>Test Types</Text>
                    <CheckboxGroup value={refusalTests} onChange={(v) => setRefusalTests(v as string[])}>
                      <SimpleGrid columns={{ base: 1, md: 2 }} spacing={2}>
                        {REFUSAL_TESTS.map((t) => (
                          <Checkbox key={t.id} value={t.id}>
                            <Text fontSize="sm">{t.label}</Text>
                          </Checkbox>
                        ))}
                      </SimpleGrid>
                    </CheckboxGroup>
                  </Box>

                  <FormControl>
                    <FormLabel>Test Prompt (harmful request)</FormLabel>
                    <Textarea
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder="Enter a harmful request to test refusal consistency..."
                      rows={3}
                    />
                  </FormControl>

                  <Button
                    colorScheme="green"
                    onClick={() => refusalMutation.mutate({ model, prompt, harm_category: harmCategory, test_types: refusalTests })}
                    isLoading={refusalMutation.isPending}
                    size="lg"
                  >
                    Run Refusal Tests
                  </Button>
                </VStack>
              </CardBody>
            </Card>
            {renderTestResults(refusalMutation.data)}
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Box>
  );
}
