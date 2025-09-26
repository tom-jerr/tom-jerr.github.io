# Implement DDL for Vector Type

## Implement CREATE TAG for Vector Type

### Thrift: Modify ColumnTypeDef

- Use `type_lenght` to indicate the dimension of the vector type.

```cpp
struct ColumnTypeDef {
    1: required common.PropertyType    type,
    // type_length is valid for fixed_string type and vector type for dimension
    2: optional i16                    type_length = 0,
    // geo_shape is valid for geography type
    3: optional GeoShape               geo_shape,
}
```

### Graphd: Create Tag Validator

We should distinguish the vector columns and other columns from parser.

### Metad: Modify Processor

Add the solution to handle the vector columns from `CreateTagReq.schema`.
