// Everything personal lives here. Edit this file to update the site's profile, links and projects.
export const site = {
  name: 'Ishwar Jangid',
  firstName: 'Ishwar',
  tagline: 'systems, machine learning, and GPUs. learning in public.',
  location: 'Delhi, India',
  email: 'iisudrj11@gmail.com',
  bio: [
    'I am a software engineer who builds production AI systems: agents, retrieval pipelines and the data platforms underneath them.',
    'I learn by writing things down properly. This site is where those notes become long-form, first-principles books on GPU programming, training LLMs from scratch, retrieval-augmented generation and ML systems, each with code you can run and numbers you can measure.',
  ],
  links: {
    github: 'https://github.com/ishwar6',
    youtube: 'https://www.youtube.com/c/IshwarJangid',
    x: 'https://x.com/IshwarSJangid',
  },
}

export type Project = { name: string; description: string; url: string; language: string; tags: string[] }

export const projects: Project[] = [
  {
    name: 'Django REST Framework course',
    description: 'The full DRF course from my YouTube series: serializers, viewsets, auth, permissions and testing, arranged lesson by lesson.',
    url: 'https://github.com/ishwar6/Django-Rest-Framework',
    language: 'Python',
    tags: ['Django', 'REST', 'Teaching'],
  },
  {
    name: 'Django CI/CD with Jenkins',
    description: 'A complete continuous-integration and delivery pipeline for a Django project, built on Jenkins.',
    url: 'https://github.com/ishwar6/django_ci_cd',
    language: 'Python',
    tags: ['CI/CD', 'Jenkins', 'DevOps'],
  },
  {
    name: 'Django on ECS with Terraform',
    description: 'Production-ready deployment package: Django, Postgres and Jenkins on AWS ECS, provisioned end to end with Terraform.',
    url: 'https://github.com/ishwar6/django-terraform-ecs',
    language: 'HCL',
    tags: ['Terraform', 'AWS', 'Infrastructure'],
  },
  {
    name: 'KST Learning Path',
    description: 'An adaptive learning engine based on Knowledge Space Theory: assesses what a student knows and routes them to what they are ready to learn next.',
    url: 'https://github.com/ishwar6/KST-Learning-Path',
    language: 'Python',
    tags: ['EdTech', 'Adaptive learning'],
  },
  {
    name: 'RAG pipeline with agents',
    description: 'A retrieval-augmented generation pipeline orchestrated by tool-using agents.',
    url: 'https://github.com/ishwar6/rag-pipeline-with-agents',
    language: 'Python',
    tags: ['RAG', 'Agents', 'LLMs'],
  },
  {
    name: 'Design patterns in Python',
    description: 'The classic Gang-of-Four patterns, each with a small, idiomatic Python example.',
    url: 'https://github.com/ishwar6/design_patterns',
    language: 'Python',
    tags: ['Software design'],
  },
]
