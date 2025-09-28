## What type of PR is this?

- [ ] bug
- [x] feature
- [ ] enhancement

## What problem(s) does this PR solve?

#### Issue(s) number:

Support vector value type and data type #6042
Implement the vector value type and serialization and deserialization for it #6067
Implement the property type and serialization and deserialization for it. #6071

#### Description:

**Feature Request: Add Data Type Specification for Vector Values**

1. We need a new value type VECTOR, like List, Map, etc.. This type is for vector data.
2. VECTOR type has original vector data and from original data we can get dimension of a vector.

**Feature Request: Add Property Type for Vector Value Type**

1. We need a new property type for VECTOR value type.
2. We also need to design the way to serialize and deserialize vector value type.

## How do you solve it?

1. Implement vector value type and do the unit test for vector type
   - Add new value type: VECTOR
   - Add serialization and deserialization method for vector type.
2. Add new property type, which value type is VECTOR
3. Add new pattern for vector value type in common.thrift

## Special notes for your reviewer, ex. impact of this fix, design document, etc:

### Vector Value Type

Vector type is consist of original vector data.
We also offered a dim() method to get the dimension of vector data, which is from vector data size.

![image](https://github.com/user-attachments/assets/abe33e6d-de50-4513-89e8-8876f1a592be)

### Vector Property Type

Add new property type, which value type is VECTOR
Add new pattern for vector value type in common.thrift

## Checklist:

Tests:

- [x] Unit test(positive and negative cases)
- [ ] Function test
- [ ] Performance test
- [ ] N/A

TCK Tests:

![image](https://github.com/user-attachments/assets/2f85234d-24dc-46c1-bfb0-724f3724e1f7)

Affects:

- [ ] Documentation affected (Please add the label if documentation needs to be modified.)
- [ ] Incompatibility (If it breaks the compatibility, please describe it and add the label.）
- [ ] If it's needed to cherry-pick (If cherry-pick to some branches is required, please label the destination version(s).)
- [ ] Performance impacted: Consumes more CPU/Memory

## Release notes:

Please confirm whether to be reflected in release notes and how to describe:

> ex. Fixed the bug .....
