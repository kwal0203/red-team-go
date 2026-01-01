import React from 'react';
import {
  Box,
  VStack,
  Link,
  Text,
  Divider,
  useColorModeValue,
} from '@chakra-ui/react';
import { Link as RouterLink, useLocation } from 'react-router-dom';

interface NavItemProps {
  to: string;
  children: React.ReactNode;
}

interface NavSectionProps {
  title: string;
  children: React.ReactNode;
}

export default function Sidebar() {
  const location = useLocation();
  const activeBg = useColorModeValue('blue.50', 'blue.900');
  const hoverBg = useColorModeValue('gray.100', 'gray.700');
  const sectionColor = useColorModeValue('gray.500', 'gray.400');

  const NavItem = ({ to, children }: NavItemProps) => {
    const isActive = location.pathname === to;
    return (
      <Link
        as={RouterLink}
        to={to}
        w="full"
        p={2}
        pl={4}
        fontSize="sm"
        borderRadius="md"
        _hover={{ bg: hoverBg, textDecoration: 'none' }}
        bg={isActive ? activeBg : 'transparent'}
        fontWeight={isActive ? 'semibold' : 'normal'}
      >
        {children}
      </Link>
    );
  };

  const NavSection = ({ title, children }: NavSectionProps) => (
    <Box>
      <Text
        fontSize="xs"
        fontWeight="bold"
        color={sectionColor}
        textTransform="uppercase"
        letterSpacing="wider"
        mb={2}
        px={2}
      >
        {title}
      </Text>
      <VStack spacing={1} align="stretch">
        {children}
      </VStack>
    </Box>
  );

  return (
    <Box
      as="nav"
      pos="sticky"
      top="4rem"
      w="240px"
      h="calc(100vh - 4rem)"
      pt={5}
      px={3}
      overflowY="auto"
      bg={useColorModeValue('white', 'gray.900')}
      borderRight="1px"
      borderRightColor={useColorModeValue('gray.200', 'gray.700')}
    >
      <VStack spacing={4} align="stretch">
        <NavItem to="/">Dashboard</NavItem>

        <Divider />

        <NavSection title="Detection">
          <NavItem to="/realtime">Real-time Analysis</NavItem>
          <NavItem to="/toxicity-batch">Toxicity Batch</NavItem>
          <NavItem to="/bias-batch">Bias Batch</NavItem>
        </NavSection>

        <NavSection title="Safety">
          <NavItem to="/guardrails">Guardrails</NavItem>
          <NavItem to="/adversarial">Adversarial Testing</NavItem>
        </NavSection>

        <NavSection title="Evaluation">
          <NavItem to="/benchmarks">Stereotype Benchmarks</NavItem>
          <NavItem to="/reliability">Reliability Testing</NavItem>
          <NavItem to="/privacy">Privacy Testing</NavItem>
          <NavItem to="/hallucination">Hallucination Detection</NavItem>
        </NavSection>
      </VStack>
    </Box>
  );
}
