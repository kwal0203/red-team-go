import React from 'react';
import {
  Box,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Text,
  Alert,
  AlertIcon,
  Card,
  CardBody,
  CardHeader,
  Heading,
  VStack,
  Badge,
  useColorModeValue,
} from '@chakra-ui/react';
import { useQuery } from '@tanstack/react-query';
import apiClient from '../api/client';
import { HEALTH_CHECK_INTERVAL } from '../api/constants';

interface StatCardProps {
  title: string;
  stat: string;
  helpText: string;
  status?: 'success' | 'warning' | 'error' | 'info';
}

function StatCard(props: StatCardProps) {
  const { title, stat, helpText, status = 'info' } = props;
  const statusColors = {
    success: 'green.500',
    warning: 'yellow.500',
    error: 'red.500',
    info: 'blue.500',
  };

  return (
    <Stat
      px={{ base: 4, md: 8 }}
      py={'5'}
      shadow={'xl'}
      border={'1px solid'}
      borderColor={useColorModeValue('gray.200', 'gray.500')}
      rounded={'lg'}
      bg={useColorModeValue('white', 'gray.700')}
    >
      <StatLabel fontWeight={'medium'} isTruncated>
        {title}
      </StatLabel>
      <StatNumber fontSize={'2xl'} fontWeight={'medium'} color={statusColors[status]}>
        {stat}
      </StatNumber>
      <StatHelpText>{helpText}</StatHelpText>
    </Stat>
  );
}

const FEATURES = [
  { name: 'Toxicity Detection', endpoints: ['/toxicity-detection-batch', '/toxicity-detection-realtime'] },
  { name: 'Bias Detection', endpoints: ['/bias-detection-batch', '/bias-detection-realtime'] },
  { name: 'Guardrails', endpoints: ['/evaluate/guardrails', '/protect/guardrails'] },
  { name: 'Adversarial Testing', endpoints: ['/adversarial-robustness', '/generate-adversarial-prompts'] },
  { name: 'Stereotype Benchmarks', endpoints: ['/stereotype-benchmark'] },
  { name: 'Reliability Testing', endpoints: ['/consistency-reliability', '/misinformation-factuality', '/refusal-consistency'] },
  { name: 'Privacy Red Team', endpoints: ['/privacy-redteam'] },
  { name: 'Hallucination Detection', endpoints: ['/hallucination-confidence'] },
];

export default function Dashboard() {
  const { data: healthData, isLoading, isError } = useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      const { data } = await apiClient.get('/health');
      return data;
    },
    retry: 1,
    refetchInterval: HEALTH_CHECK_INTERVAL,
  });

  const getSystemStatus = () => {
    if (isLoading) return { stat: 'Checking...', status: 'info' as const };
    if (isError) return { stat: 'Offline', status: 'error' as const };
    return { stat: healthData?.status || 'Healthy', status: 'success' as const };
  };

  const systemStatus = getSystemStatus();

  return (
    <Box maxW="7xl" mx={'auto'} pt={5} px={{ base: 2, sm: 12, md: 17 }}>
      <Text fontSize="2xl" fontWeight="bold" mb={2}>
        RedTeamGO Dashboard
      </Text>
      <Text color="gray.600" mb={8}>
        AI Red Teaming and Safety Evaluation Platform
      </Text>

      {isError && (
        <Alert status="warning" mb={6} borderRadius="md">
          <AlertIcon />
          Backend not connected. Start the server with: uv run uvicorn main:app --reload --port 8000
        </Alert>
      )}

      <SimpleGrid columns={{ base: 1, md: 3 }} spacing={{ base: 5, lg: 8 }} mb={8}>
        <StatCard
          title={'Backend Status'}
          stat={systemStatus.stat}
          helpText={'API server connection'}
          status={systemStatus.status}
        />
        <StatCard
          title={'Endpoints'}
          stat={'15'}
          helpText={'Available API endpoints'}
          status="success"
        />
        <StatCard
          title={'Test Coverage'}
          stat={'503'}
          helpText={'Passing tests'}
          status="success"
        />
      </SimpleGrid>

      <Card>
        <CardHeader>
          <Heading size="md">Available Features</Heading>
        </CardHeader>
        <CardBody>
          <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={4}>
            {FEATURES.map((feature) => (
              <Box key={feature.name} p={3} borderWidth={1} borderRadius="md">
                <Text fontWeight="semibold" mb={2}>{feature.name}</Text>
                <VStack align="start" spacing={1}>
                  {feature.endpoints.map((ep) => (
                    <Badge key={ep} colorScheme="blue" fontSize="xs">
                      {ep}
                    </Badge>
                  ))}
                </VStack>
              </Box>
            ))}
          </SimpleGrid>
        </CardBody>
      </Card>
    </Box>
  );
}
