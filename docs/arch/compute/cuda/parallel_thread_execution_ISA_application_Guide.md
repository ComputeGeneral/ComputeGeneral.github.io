
# Parallel Thread Execution ISA
[toc]


## 8. Memory Consistency Model

### 8.2 Memory Operation
- Data type
- Address operand
    - Memory address/address : indicate a virtual address.
    - Memory location : indicate a physical memory location.
- Operation  type
    - Read
    - Write
    - Atomic (Read-Modify-Write)

**Overlap**
    complete overlap
    partial overlap
**Aliases**
    Two distinct virtual addresses are said to be aliases if they map to the same memory location.
**Vector Data-types**
The memory consistency model relates operations executed on memory locations with scalar data-types, which have a maximum size and alignment of 64 bits

Memory operations with a vector data-type are modelled as a set of equivalent memory operations with a scalar datatype, executed in an unspecified order on the elements in the vector.

**Packed Data-types**
The packed data-type .f16x2 consists of two .f16 values accessed in adjacent memory locations. Memory operations on the packed data-type .f16x2 are modelled as a pair of equivalent memory operations with a scalar data-type .f16, executed in an unspecified order on each element of the packed data.

### 8.3 State Spaces

### 8.4 Operation Types

|Operation type|Instruction/Operation|
|----          |----                 |
|Atomic Operation|`atom` or `red` instruction|
|Read Operation|All variants of `ld` instruction and `atom` instruction but not `red` instruction|
|Write Operation|All variants of `st` instruction and `atom` operations if the result in a write|
|Memory Operation|A *Read* or *Write* Operation|
|volatile Operation|An instruction with `.volatile` qualifier|
|Acquire Operation|A *memory* operation with `.acquire` or `.acq_rel` qualifier|
|Release Operation|A *memory* operation with `.release` or `.acq_rel` qualifier|
|Memory Fence Operation|A `membar`, `fence.sc` or `fence.acq_rel` instruction|
|Proxy Fence Operation|A `fence.proxy` or `membar.proxy` instruction|
|strong Operation|A memory fence operation, or a memory operation with a `.relaxed`, `.acquire`, `.release`, `.acq_rel` or `.volatile` qualifier.|
|Weak Operation|An `ld` or `st` instruction with a `.weak` qualifier|
|Synchronizing Operation|A `bar` instruction, *fence* Operation, *release* Operation or *acquire* Operation|

### 8.5 Scope
**Each strong operation must specify a ***scope*****, which is the set of threads that may interact directly with that operation and establish any of the relations described in the memory consistency model.

|Scope|Description|
|---  |---        |
|`.cta`| |
|`.gpu`| |
|`.sys`| |

### 8.6 Proxy
A memory proxy, or a proxy is an abstract label applied to a method of memory access. When two memory operations use distinct methods of memory access, they are said to be different proxies.