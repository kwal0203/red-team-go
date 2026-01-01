import React, { useState } from 'react';
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Select,
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
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Input,
} from '@chakra-ui/react';
import { useMutation } from '@tanstack/react-query';
import apiClient from '../api/client';
import { Model } from '../api/types';

const BENCHMARKS = [
  { id: 'stereoset', label: 'StereoSet', description: 'Measures stereotypical associations in model predictions' },
  { id: 'crows_pairs', label: 'CrowS-Pairs', description: 'Crowdsourced stereotype pairs evaluation' },
  { id: 'bbq', label: 'BBQ', description: 'Bias Benchmark for Question Answering' },
];

export default function Benchmarks() {
  const toast = useToast();
  const [model, setModel] = useState<Model>({
    name: 'openai:gpt-4',
    description: 'OpenAI GPT-4 for stereotype benchmarks',
  });
  const [benchmark, setBenchmark] = useState('stereoset');
  const [numSamples, setNumSamples] = useState(10);

  const benchmarkMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await apiClient.post('/stereotype-benchmark', data);
      return response.data;
    },
    onSuccess: () => {
      toast({
        title: 'Benchmark Complete',
        description: 'Stereotype benchmark has been completed.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    },
    onError: (error: any) => {
      toast({
        title: 'Error',
        description: error.response?.data?.detail || 'Failed to run benchmark.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    },
  });

  const handleRunBenchmark = () => {
    benchmarkMutation.mutate({
      model,
      benchmark,
      num_samples: numSamples,
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
        Stereotype Benchmarks
      </Text>
      <Text color="gray.600" mb={8}>
        Evaluate model fairness using standard bias benchmarks (StereoSet, CrowS-Pairs, BBQ).
      </Text>

      <Card mb={8}>
        <CardHeader>
          <Heading size="md">Benchmark Configuration</Heading>
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
              <FormControl flex={1}>
                <FormLabel>Samples</FormLabel>
                <NumberInput
                  value={numSamples}
                  onChange={(_, val) => setNumSamples(val || 10)}
                  min={5}
                  max={100}
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>
            </HStack>

            {/* Benchmark Selection */}
            <Box>
              <Text fontWeight="semibold" mb={3}>Select Benchmark</Text>
              <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                {BENCHMARKS.map((b) => (
                  <Card
                    key={b.id}
                    cursor="pointer"
                    onClick={() => setBenchmark(b.id)}
                    borderColor={benchmark === b.id ? 'teal.500' : 'gray.200'}
                    borderWidth={2}
                    _hover={{ borderColor: 'teal.300' }}
                  >
                    <CardBody py={4}>
                      <Text fontWeight="semibold">{b.label}</Text>
                      <Text fontSize="sm" color="gray.500">{b.description}</Text>
                    </CardBody>
                  </Card>
                ))}
              </SimpleGrid>
            </Box>

            <Button
              colorScheme="teal"
              onClick={handleRunBenchmark}
              isLoading={benchmarkMutation.isPending}
              loadingText="Running..."
              size="lg"
            >
              Run Benchmark
            </Button>
          </VStack>
        </CardBody>
      </Card>

      {/* Results */}
      {benchmarkMutation.data && (
        <Card>
          <CardHeader>
            <HStack justify="space-between">
              <Heading size="md">Benchmark Results</Heading>
              {benchmarkMutation.data.overall_grade && (
                <Badge fontSize="xl" colorScheme={getGradeColor(benchmarkMutation.data.overall_grade)}>
                  Grade: {benchmarkMutation.data.overall_grade}
                </Badge>
              )}
            </HStack>
          </CardHeader>
          <CardBody>
            {benchmarkMutation.data.metrics ? (
              <VStack spacing={6} align="stretch">
                <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                  <Stat>
                    <StatLabel>Stereotype Score</StatLabel>
                    <StatNumber>{(benchmarkMutation.data.metrics.stereotype_score * 100).toFixed(1)}%</StatNumber>
                    <Progress
                      value={benchmarkMutation.data.metrics.stereotype_score * 100}
                      colorScheme={benchmarkMutation.data.metrics.stereotype_score < 0.5 ? 'green' : 'red'}
                      size="sm"
                      mt={2}
                    />
                    <StatHelpText>Lower is better</StatHelpText>
                  </Stat>
                  <Stat>
                    <StatLabel>Anti-Stereotype Score</StatLabel>
                    <StatNumber>{(benchmarkMutation.data.metrics.anti_stereotype_score * 100).toFixed(1)}%</StatNumber>
                    <Progress
                      value={benchmarkMutation.data.metrics.anti_stereotype_score * 100}
                      colorScheme="blue"
                      size="sm"
                      mt={2}
                    />
                  </Stat>
                  <Stat>
                    <StatLabel>Neutral Score</StatLabel>
                    <StatNumber>{(benchmarkMutation.data.metrics.neutral_score * 100).toFixed(1)}%</StatNumber>
                    <Progress
                      value={benchmarkMutation.data.metrics.neutral_score * 100}
                      colorScheme="green"
                      size="sm"
                      mt={2}
                    />
                    <StatHelpText>Higher is better</StatHelpText>
                  </Stat>
                </SimpleGrid>

                {benchmarkMutation.data.samples && (
                  <Box>
                    <Text fontWeight="semibold" mb={3}>Sample Results</Text>
                    <Table size="sm" variant="simple">
                      <Thead>
                        <Tr>
                          <Th>Context</Th>
                          <Th>Choice</Th>
                          <Th>Type</Th>
                        </Tr>
                      </Thead>
                      <Tbody>
                        {benchmarkMutation.data.samples.slice(0, 10).map((s: any, i: number) => (
                          <Tr key={i}>
                            <Td fontSize="sm" maxW="300px" isTruncated>{s.context}</Td>
                            <Td fontSize="sm">{s.choice}</Td>
                            <Td>
                              <Badge colorScheme={s.type === 'stereotype' ? 'red' : s.type === 'anti_stereotype' ? 'blue' : 'green'}>
                                {s.type}
                              </Badge>
                            </Td>
                          </Tr>
                        ))}
                      </Tbody>
                    </Table>
                  </Box>
                )}
              </VStack>
            ) : (
              <Box as="pre" fontSize="sm" whiteSpace="pre-wrap" bg="gray.50" p={4} borderRadius="md">
                {JSON.stringify(benchmarkMutation.data, null, 2)}
              </Box>
            )}
          </CardBody>
        </Card>
      )}
    </Box>
  );
}
